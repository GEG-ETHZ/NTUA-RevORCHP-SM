"""Run src/revorchp_sm/reversible_design_ann_vopt6_wrapper.py from the repository directory.

The code now lives in the revorchp_sm package. This file keeps the commands in
README.md working without installing the package.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from revorchp_sm.reversible_design_ann_vopt6_wrapper import *  # noqa: E402,F401,F403
from revorchp_sm.reversible_design_ann_vopt6_wrapper import main  # noqa: E402

if __name__ == "__main__":
    main()
