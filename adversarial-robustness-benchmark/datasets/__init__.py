"""Dataset loading and class-balanced sampling.

NOTE: this package is named ``datasets`` to match the project brief's repo
layout. That shadows the HuggingFace ``datasets`` library. The dataset build
deliberately uses ``pyarrow`` to read parquet directly, so HuggingFace
``datasets`` is never imported and there is no conflict. If a future module
needs HuggingFace ``datasets``, import it before adding the repo root to
``sys.path``.
"""
