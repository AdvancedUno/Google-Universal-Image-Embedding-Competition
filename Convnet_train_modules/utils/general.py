import os
import pickle
import json
import logging
import pkg_resources as pkg
from pathlib import Path


# check python
# check version
# check requirements


def set_logging(name=None):
    rank = int(os.getenv('RANK', -1))
    level = logging.INFO if rank in {-1, 0} else logging.WARNING
    log = logging.getLogger(name)
    log.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.setLevel(level)
    log.addHandler(handler)
    
set_logging()
LOGGER = logging.getLogger("text-classification")


def print_config(config):
    info = "Running with the following configs:\n"
    print("\n" + info + "\n")
    for k, v in config.items():
        info += f"\t{k} : {str(v)}\n"


def load_pickle(input_path):
    with open(str(input_path), 'rb') as file_input:
        data = pickle.load(file_input)
    return data


def save_pickle(data, output_path):
    with open(str(output_path), 'wb') as file_output:
        pickle.dump(data, file_output)


def load_json(input_path):
    with open(str(input_path), 'r') as file_input:
        data = json.load(file_input)
    return data


def save_json(data, output_path):
    with open(str(output_path), 'w') as file_output:
        json.dump(data, file_output)


def file_size(path):
    mb = 1 << 20
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / mb
    elif path.is_dir():
        return sum([file.stat().st_size for file in path.glob("**/*") if file.is_file()]) / mb 
    else:
        return 0.0


def increment_path(path, exist_ok=False, sep='', mkdir=True):
    path = Path(path)
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'
            if not os.path.exists(p):
                break
        path = Path(p)
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    return path


def colorstr(*input):
    # colorstr('blue', 'hello world') https://en.wikipedia.org/wiki/ANSI_escape_code
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0]) 
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'
    }
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']
    