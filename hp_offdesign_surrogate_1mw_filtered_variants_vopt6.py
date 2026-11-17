"""Run src/revorchp_sm/hp_offdesign_surrogate_1mw_filtered_variants_vopt6.py from the repository directory.

The code now lives in the revorchp_sm package. This file keeps the commands in
README.md working without installing the package.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from revorchp_sm.hp_offdesign_surrogate_1mw_filtered_variants_vopt6 import *  # noqa: E402,F401,F403
from revorchp_sm.hp_offdesign_surrogate_1mw_filtered_variants_vopt6 import main  # noqa: E402

if __name__ == "__main__":
    main()
