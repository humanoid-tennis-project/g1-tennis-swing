"""Location of external checkouts and caches, configured by environment variables.

Nothing here is machine specific. Defaults assume the layout described in the
README (third-party checkouts under ``<repo>/upstream/``); override with:

  ADAPT_ROOT   checkout of noitom-robotics/AdaPT (provides mjlab + rsl_rl_lib)
  GMR_ROOT     checkout of YanjieZe/GMR
  TENNIS_CACHE writable ASCII-only cache directory (Warp kernels, test scratch)
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ADAPT_ROOT = Path(os.environ.get("ADAPT_ROOT", REPO / "upstream" / "AdaPT"))
GMR_ROOT = Path(os.environ.get("GMR_ROOT", REPO / "upstream" / "GMR"))
CACHE = Path(os.environ.get("TENNIS_CACHE", REPO / ".cache"))
G1_XML_DIR = ADAPT_ROOT / "src/mjlab/asset_zoo/robots/unitree_g1/xmls"
RACKET_MOUNT = REPO / "configs" / "racket_mount.json"
