from __future__ import annotations

from codemint.loaders.detect import detect_loader
from codemint.loaders.merged import MergedFileLoader
from codemint.loaders.split import SplitFileLoader

__all__ = ["MergedFileLoader", "SplitFileLoader", "detect_loader"]
