"""
radc_delphi -- COMPATIBILITY SHIM, not the upstream package of the same name.

`figure2/` is copied near-verbatim from YidanSunResearchLab/delphi-fine-tuning, where it imports
`radc_delphi.{vocab,engine,batching}`. That repo is a fork of Delphi with its own model, its own
tokenizer and its own 50/56-token RADC vocabulary. THIS project is the upstream gerstung-lab
Delphi (`../delphi`) driving the 129-token ROSMAP tokenization built by `../tokenization`.

Keeping the package NAME means the ported figure code needs almost no edits and can be diffed
against upstream when it moves. Every adaptation to our model and our vocabulary is confined to
these three modules:

    vocab.py     our labels.csv, resolved into the id families the figure asks for
    batching.py  our .bin layout (pid, age_days, disk_token) and the disk -> model +1 shift
    engine.py    our checkpoint, and a rollout sampler with the semantics the figure requires

See ../README.md for the full list of what is exact and what is a substitution.
"""
from . import vocab  # noqa: F401
