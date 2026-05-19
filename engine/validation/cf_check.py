"""CLI shim — real logic in sat_tile_stack.cf_check."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from sat_tile_stack.cf_check import main

if __name__ == "__main__":
    main()
