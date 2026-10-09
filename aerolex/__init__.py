"""AeroLex — a production-grade, from-first-principles RAG pipeline over
airline passenger-rights law (EU261, US DOT 14 CFR 260, Delta Contracts of Carriage).

This package is the workshop's *reference implementation* ("answer key"). The
progressive notebook imports from here and asks participants to fill small TODOs.
"""
import os as _os

# Windows: let MKL (NumPy) and Torch load their OpenMP runtimes together instead
# of aborting/handing the process. Harmless elsewhere; must be set before Torch.
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "0.1.0"
