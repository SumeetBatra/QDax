import logging
import wandb
import os
import glob
import json
from attrdict import AttrDict
from colorlog import ColoredFormatter

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = ColoredFormatter(
    "%(log_color)s[%(asctime)s][%(process)05d] %(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'white,bold',
        'INFOV': 'cyan,bold',
        'WARNING': 'yellow',
        'ERROR': 'red,bold',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)
ch.setFormatter(formatter)

fh = logging.FileHandler('logs/log.txt')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)

log = logging.getLogger('rl')
log.setLevel(logging.DEBUG)
log.handlers = []  # No duplicated handlers
log.propagate = False  # workaround for duplicated logs in ipython
log.addHandler(ch)
log.addHandler(fh)


def config_wandb(project, entity, group, run_name, **kwargs):
    # wandb initialization
    wandb.init(project=project, entity=entity, group=group, name=run_name)
    cfg = kwargs.get('cfg', None)
    if cfg is None:
        cfg = {}
        for key, val in kwargs.items():
            cfg[key] = val
    wandb.config.update(cfg)


def save_cfg(dir, cfg):
    def to_dict(cfg):
        if isinstance(cfg, AttrDict):
            cfg = dict(cfg)
    filename = 'cfg.json'
    fp = os.path.join(dir, filename)
    with open(fp, 'w') as f:
        json.dump(cfg, f, default=to_dict, indent=4)


def get_checkpoints(checkpoints_dir):
    checkpoints = glob.glob(os.path.join(checkpoints_dir, 'cp_*'))
    return sorted(checkpoints)
