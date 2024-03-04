"""This file is used as the central import for the lightworks module."""

from pathlib import Path
import sys
path = Path(__file__).parent.parent
sys.path.append(path)

import lightworks