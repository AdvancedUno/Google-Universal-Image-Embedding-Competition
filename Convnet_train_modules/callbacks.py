import os
import time
import shutil
import subprocess
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from utils.general import LOGGER, load_json, save_json, colorstr

CO_CODE = ("blue", "bold")


class Callback(object):
    def reset(self):
        raise NotImplementedError()
    
    def epoch_step(self):
        raise NotImplementedError()


class EarlyStopping(Callback):
    def __init__(
        self, min_delta=0, patience=10,
        mode='max', monitor='val_accuracy'
    ):
        super().__init__()
        assert mode in ['min', 'max']

        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta

        if mode == 'min':
            self.monitor_op = np.less
        else:
            self.monitor_op = np.greater

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

        self.reset()

    def reset(self):
        self.wait = 0
        self.stop_training = False
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def epoch_step(self, current):
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                LOGGER.info(colorstr(*CO_CODE, f" {self.patience} epochs with no improvement in {self.monitor}. Stopping training."))
                self.stop_training = True


class ModelCheckpoint(Callback):
    def __init__(
        self, checkpoint_dir, 
        monitor='val_accuracy', 
        mode='max'
    ):
        super().__init__()
        
        self.base_path = checkpoint_dir
        self.monitor = monitor
        self.model_name = "best.pt"
        
        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater

        self.reset()

    def reset(self):
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def epoch_step(self, current, ckpt):
        if self.monitor_op(current, self.best):
            ckpt['best'] = current
            best_path = self.base_path / self.model_name
            torch.save(ckpt, str(best_path))
            LOGGER.info(colorstr(*CO_CODE, f" \nEpoch {ckpt['epoch']}: {self.monitor} improved from {self.best:.5f} to {current:.5f}"))
            self.best = current


class TrainingMonitor(Callback):
    def __init__(self, file_dir, add_test=False):
        super().__init__()
        
        self.H = {}
        self.file_dir = file_dir
        self.add_test = add_test
        self.json_path = file_dir / "training_monitor.json"
        
    def reset(self, start_at):
        if start_at > 0:
            if self.json_path is not None:
                if self.json_path.exists():
                    self.H = load_json(self.json_path)
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:start_at]

    def epoch_step(self, logs={}):
        for (k, v) in logs.items():
            l = self.H.get(k, [])
            if not isinstance(v, np.float):
                v = round(float(v), 4)
            l.append(v)
            self.H[k] = l

        save_json(data=self.H, output_path=self.json_path)

        if len(self.H["loss"]) == 1:
            self.paths = {key: self.file_dir / f'{key.upper()}' for key in self.H.keys()}

        if len(self.H["loss"]) > 1:
            keys = [key for key, _ in self.H.items() if '_' not in key]
            for key in keys:
                N = np.arange(0, len(self.H[key]))
                plt.style.use("ggplot")
                plt.figure()
                plt.plot(N, self.H[key], label=f"train_{key}")
                plt.plot(N, self.H[f"val_{key}"], label=f"val_{key}")
                if self.add_test:
                    plt.plot(N, self.H[f"test_{key}"], label=f"test_{key}")
                plt.legend()
                plt.xlabel("Epoch #")
                plt.ylabel(key)
                plt.title(f"Training {key} [Epoch {len(self.H[key])}]")
                plt.savefig(str(self.paths[key]))
                plt.close()


class CSVLogger(Callback):
    def __init__(self, file_dir):
        super().__init__()
        self.csv_path = file_dir / "result.csv"
        self.reset()

    def reset(self):
        self.results = pd.DataFrame()
    
    def epoch_step(self, epoch, logs):
        df = pd.DataFrame(logs, index=[0])
        df['epoch'] = epoch
        self.results = self.results.append(df, ignore_index=True)
        self.results.to_csv(self.csv_path, index=False)
            

class ProgressBar(object):
    def __init__(self, n_total, width=30, desc='Training'):
        self.width = width
        self.n_total = n_total
        self.start_time = time.time()
        self.desc = desc

    def __call__(self, step, info={}):
        now = time.time()
        current = step + 1
        recv_per = current / self.n_total
        bar = f'[{self.desc}] {current}/{self.n_total} ['
        if recv_per >= 1:
            recv_per = 1
        prog_width = int(self.width * recv_per)
        if prog_width > 0:
            bar += '=' * (prog_width - 1)
            if current< self.n_total:
                bar += ">"
            else:
                bar += '='
        bar += '.' * (self.width - prog_width)
        bar += ']'
        show_bar = f"\r{bar}"
        time_per_unit = (now - self.start_time) / current
        if current < self.n_total:
            eta = time_per_unit * (self.n_total - current)
            if eta > 3600:
                eta_format = ('%d:%02d:%02d' %
                              (eta // 3600, (eta % 3600) // 60, eta % 60))
            elif eta > 60:
                eta_format = '%d:%02d' % (eta // 60, eta % 60)
            else:
                eta_format = '%ds' % eta
            time_info = f' - ETA: {eta_format}'
        else:
            if time_per_unit >= 1:
                time_info = f' {time_per_unit:.1f}s/step'
            elif time_per_unit >= 1e-3:
                time_info = f' {time_per_unit * 1e3:.1f}ms/step'
            else:
                time_info = f' {time_per_unit * 1e6:.1f}us/step'

        show_bar += time_info
        if len(info) != 0:
            s = []
            for key, value in info.items():
                if not isinstance(value, str):
                    if 'lr' not in key:
                        s.append(f' {key}:{value:.4f}')
                    else:
                        s.append(f' {key}:{value:.7f}')
                else:
                    s.append(f' {key}:{value}')
            show_info = f'{show_bar} ' + " -".join(s)
            print(colorstr(*CO_CODE, show_info), end='')
        else:
            print(colorstr(*CO_CODE, show_bar), end='')

