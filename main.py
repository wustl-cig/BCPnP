import pytorch_lightning as pl
pl.seed_everything(1016)

from bcpnp import run

import yaml
import sys

model_dict = {
    'bcpnp': run,
}

if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        config = yaml.safe_load(f)

    model_dict[config['setting']['method']](config)
