from datetime import datetime
from copy import deepcopy
import math
import torch
from callbacks import ProgressBar
from utils.general import LOGGER, colorstr
from utils.torch_utils import AverageMeter, init_full_determinism, init_seed, time_sync , is_apex_available

if is_apex_available():
    from apex import amp

CO_CODE = ("blue", "bold")

class Trainer:
    def __init__(
        self, args, device, model, ema, use_apex,
        scaler, do_grad_scaling, autocast_context_manager,
        criterion, optimizer, scheduler,
        early_stopping, model_checkpointing,
        csv_logger, training_monitor,
        epoch_metrics, batch_metrics,  
    ):
        self.args = args
        
        self.device = device
        self.model = model
        self.ema = ema
        
        self.use_apex = use_apex
        self.scaler = scaler
        self.do_grad_scaling = do_grad_scaling
        self.autocast_context_manager = autocast_context_manager
        
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.early_stopping = early_stopping
        self.model_checkpointing = model_checkpointing
        self.csv_logger = csv_logger
        self.training_monitor = training_monitor
        
        self.epoch_metrics = epoch_metrics
        self.batch_metrics = batch_metrics
        
        self.epochs_trained = 0
        self.steps_trained_in_current_epoch = 0
    
    
    def epoch_reset(self):
        self.outputs = []
        self.targets = []
        self.result = {}
        for metric in self.epoch_metrics:
            metric.reset()
    
    
    def batch_reset(self):
        self.info = {}
        self.lrs = {}
        for metric in self.batch_metrics:
            metric.reset()
    
    
    def train_one_epoch(self, train_loader):
        pbar = ProgressBar(n_total=len(train_loader), desc='Training ')
        training_loss = AverageMeter()
        self.epoch_reset()
        for step, batch in enumerate(train_loader):
            self.batch_reset()
            self.model.train()
            
            # Prepare inputs
            batch = {k: v.to(self.device) for k, v in batch.items()}
            inputs = batch.pop('image')
            targets = batch.pop('labels')
            
            # Forward Propagation
            with torch.set_grad_enabled(True):
                with self.autocast_context_manager:
                    logits = self.model(inputs)
                    loss = self.criterion(logits, targets)
            
            if torch.cuda.device_count() > 1:
                # If DDP then average on multi-gpu parallel training
                loss = loss.mean()
                
            if self.args.gradient_accumulation_steps > 1:
                # Loss scaling by gradient_accumulation_steps
                loss = loss / self.args.gradient_accumulation_steps
            
            # Backward Progpagation
            if self.do_grad_scaling:
                self.scaler.scale(loss).backward()
            elif self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
                
            training_loss.update(loss.item(), n=1)
            
            if ((step + 1) % self.args.gradient_accumulation_steps == 0) or ((step + 1) == len(train_loader)):
                # Gradient Clipping
                if self.do_grad_scaling:
                    # AMP: gradients need unscaling
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad.clip_grad_norm_(
                    amp.master_params(self.optimizer) if self.use_apex else self.model.parameters(),
                    self.args.max_grad_norm
                )
                
                # Optimizer step
                optimizer_was_run = True
                if self.do_grad_scaling:
                    scale_before = self.scaler.get_scale()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    scale_after = self.scaler.get_scale()
                    optimizer_was_run = scale_before <= scale_after
                else:
                    self.optimizer.step()
                
                # Scheduler step
                if optimizer_was_run:
                    # Learning rate values
                    for i, x in enumerate(self.optimizer.param_groups):
                        self.lrs[f'lr{i+1}'] = x['lr']
                    self.scheduler.step()
                
                # EMAModel update
                if self.ema:
                    self.ema.update(self.model)
                    
                # Zero out gradients
                self.model.zero_grad()
                
                self.steps_trained_in_current_epoch += 1
            
            # Calculate batch metrics
            self.info['loss'] = loss.item()
            if self.batch_metrics:
                for metric in self.batch_metrics:
                    metric(logits, targets)
                    self.info[metric.name()] = metric.value()
            self.info['gpu_mem'] = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
            self.info = {**self.info, **self.lrs}
            
            # Update step
            pbar(step=step, info=self.info)
            
            self.outputs.append(logits) 
            self.targets.append(targets) 
        
        LOGGER.info(colorstr(*CO_CODE, "\n ---------- Training Result ----------"))
        self.outputs = torch.cat(self.outputs, dim=0)
        self.targets = torch.cat(self.targets, dim=0)
        self.result['loss'] = training_loss.avg
        
        # Calculate epoch metrics
        if self.epoch_metrics:
            for metric in self.epoch_metrics:
                metric(self.outputs, self.targets)
                self.result[metric.name()] = metric.value()
        
        # Empty CUDA cache
        if self.device.type != 'cpu':
            torch.cuda.empty_cache()
        
        return self.result
    
    
    def validation_step(self, validation_loader):
        pbar = ProgressBar(n_total=len(validation_loader), desc='Validation ')
        self.epoch_reset()
        for step, batch in enumerate(validation_loader):
            self.model.eval()
            
            # Prepare inputs
            batch = {k: v.to(self.device) for k, v in batch.items()}
            inputs = batch.pop('image')
            targets = batch.pop('labels')

            # Forward Propagation
            with torch.set_grad_enabled(False):
                logits = self.model(inputs)
                loss = self.criterion(logits, targets)
            self.outputs.append(logits) 
            self.targets.append(targets)
            
            # Update step
            pbar(step=step)
                   
        LOGGER.info(colorstr(*CO_CODE, "\n ---------- Validation Result ----------"))
        self.outputs = torch.cat(self.outputs, dim=0)
        self.targets = torch.cat(self.targets, dim=0)
        self.result['val_loss'] = self.criterion(self.outputs, self.targets).item()
        
        # Calculate epoch metrics
        if self.epoch_metrics:
            for metric in self.epoch_metrics:
                metric(self.outputs, self.targets)
                self.result[f'val_{metric.name()}'] = metric.value()
        
        # Empty CUDA cache
        if self.device.type != 'cpu':
            torch.cuda.empty_cache()
        
        return self.result
        
    
    def train(self, train_loader, validation_loader):
        init_full_determinism(self.args.seed) if self.args.full_determinism else init_seed(self.args.seed)
        self.model.zero_grad()
        for epoch in range(self.epochs_trained, self.args.num_epochs):
            LOGGER.info(colorstr(*CO_CODE, f"\n Epoch {epoch + 1} / {self.args.num_epochs}"))
            training_logs = self.train_one_epoch(train_loader)
            validation_logs = self.validation_step(validation_loader)            
            logs = {**training_logs, **validation_logs}
            LOGGER.info(colorstr(*CO_CODE, f'\n Epoch: {epoch + 1} - ' + "-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])))

            if self.early_stopping:
                self.early_stopping.epoch_step(logs[self.early_stopping.monitor])
                if self.early_stopping.stop_training:
                    break

            if self.model_checkpointing:
                ckpt = {
                    'epoch': epoch,
                    'model': deepcopy(self.model.state_dict()),
                    # 'ema': deepcopy(self.ema.state_dict() if self.ema else self.ema),
                    # 'updates': self.ema.updates if self.ema else self.ema,
                    # 'optimizer': self.optimizer.state_dict(),
                    'date':datetime.now().isoformat()
                }
                self.model_checkpointing.epoch_step(current=logs[self.model_checkpointing.monitor], ckpt=ckpt)

            if self.training_monitor:
                self.training_monitor.epoch_step(logs)

            if self.csv_logger:
                self.csv_logger.epoch_step(epoch, logs)