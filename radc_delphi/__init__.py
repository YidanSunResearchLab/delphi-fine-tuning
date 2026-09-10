"""radc_delphi -- a Delphi-2M-style event-sequence model for the RADC/ROSMAP cohorts.

Layout:
    vocab.py          the token table: the single source of truth for what every id means
    tokenizer.py      the three raw RADC files -> an age-ordered event stream
    splits.py         by-subject train/val/test partitioning
    build_dataset.py  CLI wrapper: tokenize -> split -> data/radc-s<seed>/
    model.py          the Delphi transformer (vocabulary-agnostic)
    batching.py       patient indexing and batch assembly for training
    engine.py         inference: next-event probabilities and trajectory sampling
    decode_codebook.py  makes the RADC codebook PDF readable (Type3 glyph-hash decoding)

Token spaces, which are easy to confuse and expensive to get wrong:
    DISK space   what lives in the .bin files (data[:, 2]). disk = model - 1.
    MODEL space  what the model sees, and what labels.csv is indexed by.
                 get_batch applies the +1 shift; 0 is Padding and 1 is No-event.
"""
