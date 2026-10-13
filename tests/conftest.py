"""Tests exercise THIS checkout's ``src``, whatever the interpreter has installed.

The package is installed editable, and an editable install points at one
checkout's ``src``. A run from any other tree (another worktree sharing the
venv, a scratch export, the arming prover's throwaway tree of HEAD) would
otherwise import that checkout's code and test it instead of its own, so the
run is green or red for the wrong tree.
"""

import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parent.parent / "src")
if sys.path[:1] != [_SRC]:
    sys.path.insert(0, _SRC)
