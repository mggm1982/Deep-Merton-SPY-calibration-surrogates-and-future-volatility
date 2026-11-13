"""Run with: python run_project.py (paths resolve from this file)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from finance_project.pipeline import main

if __name__ == "__main__":
    main()
