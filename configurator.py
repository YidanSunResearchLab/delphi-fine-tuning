"""
Configurator -- load a config file, then apply CLI overrides.

Exec'd by train.py into its own globals, so a config file is just plain `key = value`
assignments and any name it sets must already exist as a default in train.py (an unknown key
is a hard error, not a silent no-op -- that is how a typo'd hyperparameter used to be ignored).

  python train.py configs/radc_base.py
  python train.py configs/radc_base.py --wandb online
  python train.py configs/radc_base.py --max_iters 4000
  python train.py configs/radc_base.py --wandb online --batch_size 64

.py configs are exec'd (zero dependencies); .yaml/.yml need pyyaml and are imported lazily.
"""

import sys
import argparse
from ast import literal_eval

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('config', nargs='?', default=None)
parser.add_argument('--wandb', choices=['online', 'offline', 'disabled'], default='disabled',
                    help='wandb mode — disabled by default to avoid polluting the dashboard')

args, overrides = parser.parse_known_args()

# Load config (.py = zero deps, exec'd; .yaml/.yml = needs pyyaml, imported lazily)
if args.config:
    if args.config.endswith((".yaml", ".yml")):
        import yaml  # only needed for YAML configs
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
    else:            # .py config: plain `key = value` assignments, no third-party deps
        ns = {}
        with open(args.config) as f:
            exec(f.read(), ns)
        cfg = {k: v for k, v in ns.items() if not k.startswith("_")}
    for k, v in cfg.items():
        if k in globals():
            globals()[k] = v
        else:
            raise ValueError(f"Unknown config key '{k}' in {args.config}")
    print(f"Loaded config: {args.config}")

# wandb mode — CLI always wins
globals()['wandb_log'] = (args.wandb != 'disabled')   # 'online' or 'offline' both log
if args.wandb == 'offline':
    import os
    os.environ['WANDB_MODE'] = 'offline'
print(f"wandb: {args.wandb}")

# Apply remaining --key value or --key=value overrides
i = 0
while i < len(overrides):
    token = overrides[i]
    if '=' in token:
        key, val = token.lstrip('-').split('=', 1)
        i += 1
    else:
        key = token.lstrip('-')
        i += 1
        if i >= len(overrides):
            raise ValueError(f"No value provided for --{key}")
        val = overrides[i]
        i += 1

    if key not in globals():
        raise ValueError(f"Unknown config key: {key}")

    try:
        attempt = literal_eval(val)
    except (SyntaxError, ValueError):
        attempt = val

    globals()[key] = attempt
    print(f"Override: {key} = {attempt}")
