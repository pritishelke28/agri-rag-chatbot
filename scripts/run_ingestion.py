#!/usr/bin/env python
"""Entry point: python scripts/run_ingestion.py"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.ingestion.ingest import run

if __name__ == "__main__":
    run()
