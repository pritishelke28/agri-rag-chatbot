#!/usr/bin/env python
"""Entry point: python scripts/train_crop_model.py"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.crop_recommender import train

if __name__ == "__main__":
    train()
