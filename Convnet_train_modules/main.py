from enum import auto
import os
import sys
import time
import math
from tracemalloc import start
import warnings
import argparse
import contextlib
from packaging import version
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

import timm
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from model import Model
from optimization import init_optimizer_cls_and_kwargs, get_grouped_parameters
from scheduler import init_scheduler
from metrics import Accuracy, F1Score
from losses import init_criterion
from callbacks import (
    ModelCheckpoint, EarlyStopping, 
    CSVLogger, TrainingMonitor
)

from train import Trainer

from dataset import create_folds, DatasetRetriever, generate_transforms, transforms_train, transforms_validation
from utils.general import colorstr, increment_path, LOGGER, save_json
from utils.torch_utils import (
    select_device, model_info, 
    init_optimal_loader_workers, 
    ModelEMA, get_world_size, is_apex_available,
    init_full_determinism, init_seed, speed_metrics
)

warnings.filterwarnings("ignore")

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
os.environ['TOKENIZERS_PARALLELISM'] = "false"
CO_CODE = ("blue", "bold")

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)

_is_torch_generator_available = False
_is_native_cuda_amp_available = False
_is_native_cpu_amp_available = False
_is_nvidia_apex_amp_available = False

if is_apex_available():
    from apex import amp
    _is_nvidia_apex_amp_available = True

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_torch_generator_available = True
    _is_native_cuda_amp_available = True

if version.parse(torch.__version__) >= version.parse("1.10"):
    _is_native_cpu_amp_available = True
    

def main(opt):
    init_full_determinism(opt.seed) if opt.full_determinism else init_seed(opt.seed)
    
    if RANK in {-1, 0}:
        LOGGER.info(colorstr(*CO_CODE, "training: " + ', '.join(f'{k}={v}' for k, v in vars(opt).items())))
        # check requirements


    # Directories
    LOGGER.info(colorstr(*CO_CODE, " Setting Directories"))
    save_dir = increment_path(Path(opt.project) / opt.experiment_name, exist_ok=False)
    (save_dir / 'logs').mkdir(parents=True, exist_ok=True)
    (save_dir / 'figure').mkdir(parents=True, exist_ok=True)
    (save_dir / 'weights').mkdir(parents=True, exist_ok=True)
    (save_dir / 'result').mkdir(parents=True, exist_ok=True)


    # Device and DDP mode
    device = select_device(opt.device)
    cuda = device.type != 'cpu'
    if LOCAL_RANK != -1:
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device("cuda", LOCAL_RANK)
        dist.init_process_group(backend='nccl' if dist.is_available() else "gloo")


    # Processor 
    dataframe = pd.read_csv(os.path.join(str(opt.data_path), 'train.csv'), low_memory=False, squeeze=True)
    dataframe, label_mapping = create_folds(dataframe)
    num_labels = len(label_mapping)
    
    
    # Model
    model = timm.create_model(opt.model_name_or_path, pretrained=True, num_classes=num_labels).to(device)
    model_info(model, image_size=opt.image_size)
    
    
    # Dataset
    train_dataset = DatasetRetriever(
        data_path = opt.data_path,
        dataframe=dataframe[dataframe['fold'] != opt.fold_id], 
        transform=transforms_train(image_size=opt.image_size),
        label_mapping=label_mapping
    )
    validation_dataset = DatasetRetriever(
        data_path = opt.data_path,
        dataframe=dataframe[dataframe['fold'] == opt.fold_id], 
        transform=transforms_validation(opt.image_size),
        label_mapping=label_mapping
    )


    # Sampler
    generator = torch.Generator()
    if opt.data_seed is None:
        seed = torch.empty((), dtype=torch.int64).random_().item()
    else:
        seed = opt.data_seed
    generator.manual_seed(seed)
    if _is_torch_generator_available:
        train_sampler = RandomSampler(data_source=train_dataset, generator=generator)
    else:
        train_sampler = RandomSampler(data_source=train_dataset)
    validation_sampler = SequentialSampler(validation_dataset)


    # Data Loader
    train_loader = DataLoader(
        train_dataset, batch_size=opt.train_batch_size, 
        drop_last=False, pin_memory=cuda, sampler=train_sampler, 
        num_workers=init_optimal_loader_workers()
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=opt.validation_batch_size, 
        drop_last=False, pin_memory=cuda, sampler=validation_sampler, 
        num_workers=init_optimal_loader_workers()
    )
    LOGGER.info(colorstr(*CO_CODE, f" DataLoaders initialized"))
    

    # Criterion
    criterion_type = opt.criterion_type if not opt.label_smoothing else "label_smoothing"
    criterion = init_criterion(criterion_type=criterion_type, label_smoothing_factor=opt.label_smoothing_factor)
    LOGGER.info(colorstr(*CO_CODE, f" {criterion_type} loss initialized"))


    # Optimizer
    grouped_parameters = get_grouped_parameters(model)
    optimizer_cls, optimizer_kwargs = init_optimizer_cls_and_kwargs(opt.optim, opt.lr, opt.weight_decay, opt.adam_beta1, opt.adam_beta2, opt.eps)
    optimizer = optimizer_cls(grouped_parameters, **optimizer_kwargs)
    LOGGER.info(colorstr(*CO_CODE, f" Optimizer {opt.optim} initialized"))    


    # Scheduler
    len_dataloader = len(train_loader)
    num_update_steps_per_epoch = max(len_dataloader // opt.gradient_accumulation_steps, 1) 
    num_examples = len(train_loader.dataset)
    max_steps = math.ceil(opt.num_epochs * num_update_steps_per_epoch)
    num_train_samples = num_examples * opt.num_epochs
    num_warmup_steps = math.ceil(max_steps * opt.warmup_ratio)
    scheduler  = init_scheduler(
        opt.lr_scheduler_type, optimizer=optimizer, 
        num_training_steps=max_steps, num_warmup_steps=num_warmup_steps
    )
    LOGGER.info(colorstr(*CO_CODE, f" Scheduler {opt.lr_scheduler_type} initialized - {max_steps} training steps, {num_warmup_steps} warmup steps"))


    # Exponential Moving Average
    ema = None
    if opt.ema:
        LOGGER.info(colorstr(*CO_CODE, f" ModelEMA initialized"))
        ema = ModelEMA(model)


    # Mixed Precision Setup
    use_apex = use_cuda_amp = use_cpu_amp = False
    if opt.fp16:
        if opt.half_precision_backend == 'auto':
            if not cuda:
                if opt.fp16:
                    raise ValueError('Tried to used fp16 but it is not available on cpu')
                elif _is_native_cpu_amp_available:
                    half_precision_backend = "cpu_amp"
                else:
                    raise ValueError("Tried to use cpu amp but native cpu amp is not available") 
            else:
                if _is_nvidia_apex_amp_available:
                    half_precision_backend = "apex"
                elif _is_native_cuda_amp_available:
                    half_precision_backend = "cuda_amp"
                else:
                    raise ValueError("Tried to use fp16 but neither cuda amp nor apex amp is available")
        else:
            half_precision_backend = opt.half_precision_backend
            
        LOGGER.info(colorstr(*CO_CODE, f" Using {half_precision_backend} half precision backend"))

    do_grad_scaling = False
    scaler = None
    autocast_context_manager = contextlib.nullcontext() if sys.version_info >= (3, 7) else contextlib.suppress()
    if half_precision_backend == "cuda_amp":
        use_cuda_amp = True
        do_grad_scaling = True
        amp_dtype = torch.float16
        scaler = torch.cuda.amp.grad_scaler.GradScaler()
        if version.parse(torch.__version__) >= version.parse("1.10"):
            autocast_context_manager = torch.cuda.amp.autocast_mode.autocast(enabled=True, dtype=amp_dtype)
        else:
            autocast_context_manager = torch.cuda.amp.autocast_mode.autocast(enabled=True)
    elif half_precision_backend == "cpu_amp":
        use_cpu_amp = True
        amp_dtype = torch.bfloat16
        autocast_context_manager = torch.cpu.amp.autocast_mode.autocast(enabled=True, dtype=amp_dtype)
    else:
        use_apex = True


    # Gradient Checkpointing 
    if opt.gradient_checkpointing and hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(enable=True)
        LOGGER.info(colorstr(*CO_CODE, f" Gradient Checkpointing initialized"))


    # Intialize mixed precision training with apex
    if use_apex:
        model, optimizer = amp.initialize(
            model, 
            optimizers=optimizer, 
            opt_level=opt.fp16_opt_level, 
            keep_batchnorm_fp32=True if opt.fp16_opt_level != 'O1' else None, 
            loss_scale='dynamic' if opt.loss_scale is None else opt.loss_scale
        )
        LOGGER.info(colorstr(*CO_CODE, f" AMP initialized with Opt level {opt.fp16_opt_level}"))


    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.info(colorstr(*CO_CODE, f" Data Parallel initialized"))
        model = torch.nn.parallel.DataParallel(model)


    # DDP mode
    if cuda and RANK != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[LOCAL_RANK], 
            output_device=LOCAL_RANK, find_unused_parameters=True
        )               
        LOGGER.info(colorstr(*CO_CODE, f" DDP initialized"))


    # Callbacks
    LOGGER.info(colorstr(*CO_CODE, f" Model Checkpointing, Early Stopping, GPU Stats Monitor, Training Monitor, CSV Logger intialized"))
    model_checkpointing = ModelCheckpoint(checkpoint_dir=(save_dir / 'weights'), monitor=opt.monitor, mode=opt.mode)
    early_stopping = EarlyStopping(min_delta=0, patience=opt.patience, monitor=opt.monitor, mode=opt.mode)
    training_monitor = TrainingMonitor((save_dir / 'figure'))
    csv_logger = CSVLogger((save_dir / 'result'))


    # Initialize Trainer
    trainer = Trainer(
        args=opt, device=device, model=model, ema=ema, use_apex=use_apex,
        scaler=scaler, do_grad_scaling=do_grad_scaling, autocast_context_manager=autocast_context_manager,
        criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        early_stopping=early_stopping, model_checkpointing=model_checkpointing,
        csv_logger=csv_logger, training_monitor=training_monitor,
        batch_metrics=[Accuracy(topK=1), F1Score(average='macro')], 
        epoch_metrics=[Accuracy(topK=1), F1Score(average='macro')]
    )


    # Train
    LOGGER.info(colorstr(*CO_CODE, " ****** Running Training ******"))
    LOGGER.info(colorstr(*CO_CODE, f" Number of examples = {num_examples}"))
    LOGGER.info(colorstr(*CO_CODE, f" Number of epochs = {opt.num_epochs}"))
    LOGGER.info(colorstr(*CO_CODE, f" Instantaneous batch size per device = {opt.train_batch_size}"))
    LOGGER.info(colorstr(*CO_CODE, f" Total train batch size (w. parallel, distributed & accumulation) = {opt.train_batch_size * opt.gradient_accumulation_steps * get_world_size()}"))
    LOGGER.info(colorstr(*CO_CODE, f" Gradient Accumulation steps = {opt.gradient_accumulation_steps}"))
    LOGGER.info(colorstr(*CO_CODE, f" Total Optimization Steps = {max_steps}"))


    # Run    
    start_time = time.time()
    trainer.train(train_loader=train_loader, validation_loader=validation_loader)
    metrics = speed_metrics(mode='train', start_time=start_time, num_samples=num_train_samples, num_steps=max_steps)
    LOGGER.info(colorstr(*CO_CODE, f'\nSpeed Metrics: ' + "- ".join([f'{key}: {value} ' for key, value in metrics.items()])))


    # Save arguments
    args = {}
    for k, v in vars(opt).items():
        if isinstance(v, Path):
            v = str(v)
        args[k] = v
    save_json(args, save_dir / 'result' / 'args.json')


def parse_opt():
    parser = argparse.ArgumentParser()

    # experiment
    parser.add_argument('--seed', default=0, type=int, help='seed value of RNG (random number generator)')
    parser.add_argument('--data_path', default=(ROOT / 'raw'), help='training data file path')
    parser.add_argument('--full_determinism', action='store_true', help='enable torch deterministic mode')
    parser.add_argument('--project', default=ROOT / 'runs', help='save outputs to project')
    parser.add_argument('--experiment_name', default='exp')
    
    # model
    parser.add_argument('--device', default='', help='cuda device or cpu')
    parser.add_argument('--model_name_or_path', default='convnext_base_in22k', help='architecture name')
    parser.add_argument('--pretrained', action="store_false", default=True, help='use imagenet checkpoint for backbone')
    parser.add_argument('--fp16', action='store_false', help='enable fp16 training')
    parser.add_argument('--half_precision_backend', default='auto', choices=["auto", "cuda_amp", "apex"], help='the backend to be used for half precision ["auto"]')
    parser.add_argument('--fp16_opt_level', default='O1', choices=['O0', 'O1', 'O2', 'O3'], help='optimization level for apex fp16')
    parser.add_argument('--ema', action="store_true", default=False, help="activate model ema")
    
    # criterion
    parser.add_argument('--criterion_type', default='ce', help='loss function to be used')
    parser.add_argument('--label_smoothing', action='store_true', help='label-smoothing on a pre-computed output')
    parser.add_argument('--label_smoothing_factor', default=0.1, type=float, help='epsilon value for label smoothing')

    # data
    parser.add_argument('--data_seed', default=None, type=int, help='seed value to use for data sampling')
    parser.add_argument('--train_batch_size', default=8, type=int, help='training batch size')
    parser.add_argument('--validation_batch_size', default=16, type=int, help='validation batch size')
    parser.add_argument('--image_size', type=int, default=384, help='image size to use for training models')

    # validation_setup
    parser.add_argument('--split_method', default='stratified_grouped', choices=['stratified_grouped', 'stratfied', 'grouped', 'kfold'], help='validation split method')
    parser.add_argument('--n_splits', type=int, default=5, help='number of splits to create')
    parser.add_argument('--fold_id', default=0, type=int, help='fold number to select for validation data')

    # optimizer
    parser.add_argument('--optim', default='sgd', help='optimizer type to initialize')
    parser.add_argument('--grouped_parameters', default='None', help='group parameters method')
    parser.add_argument('--lr', default=0.05, type=float, help='initial learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay value for parameter regularization')
    parser.add_argument('--eps', default=1e-8, type=float, help='epsilon value for optimizer')
    parser.add_argument('--adam_beta1', default=0.9, type=float, help='adam beta1 value')
    parser.add_argument('--adam_beta2', default=0.999, type=float, help='adam beta1 value')

    # scheduler
    parser.add_argument('--lr_scheduler_type', default='cosine', help='learning rate scheduler type')
    parser.add_argument('--warmup_ratio', default=0.1, type=float, help='warmup proportion of overall training steps')

    # training
    parser.add_argument('--num_epochs', default=10, type=int, help='number of training epochs')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='gradient accumulation steps to perform during training')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='gradient clipping norm value')
    parser.add_argument('--loss_scale', default=None, type=int, help='loss scaling to improve fp16 numeric stability.')  
    parser.add_argument('--gradient_checkpointing', action='store_false', help='activate gradient checkpointing if needed')  

    # callbacks
    parser.add_argument('--monitor', default='val_accuracy', help='quantity to be monitored')
    parser.add_argument('--mode', default='max', help='{"min", "max"} training will stop once the quantity monitored has stopped descreasing')
    parser.add_argument('--patience', default=4, type=int, help='number of epochs with no improvement after which training will be stopped')

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
    

