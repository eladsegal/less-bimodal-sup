import sys, os

sys.path.insert(0, os.getcwd())
import runpy

if __name__ == "__main__":
    runpy.run_module("src.eval", run_name="__main__")
