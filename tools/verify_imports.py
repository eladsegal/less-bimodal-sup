import sys, os
import glob
import traceback
import importlib
import importlib.util
import re
from tqdm import tqdm

from hydra._internal.utils import _locate


if __name__ == "__main__":
    found_errors = False

    print('Checking all ".py" files')
    for file_path in tqdm(glob.iglob("**/*.py", recursive=True)):
        if (
            os.path.realpath("setup.py") == os.path.realpath(file_path)
            or os.path.basename(os.path.dirname(file_path)) == "streamlit_apps"
        ):
            continue
        spec = importlib.util.spec_from_file_location(os.path.basename(file_path), file_path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except ModuleNotFoundError as e:
            found_errors = True
            print(f"{file_path} erroneously tried to load {e.name}")
        except ImportError as e:
            found_errors = True
            print(f"{file_path}: {e.msg}")
        except Exception as e:
            found_errors = True
            exceptiondata = traceback.format_exc().splitlines()
            print(exceptiondata[-1])
    print()

    print("Checking all imports")
    import_regex = re.compile(r"[^\"']import(?: )(?:['\"](.+)['\"])")
    for file_path in tqdm(glob.iglob("configs/**/*.json*", recursive=True)):
        with open(file_path, mode="r") as f:
            for line in f.readlines():
                line = line.strip()
                result = import_regex.search(line)
                if result is not None:
                    imported_file = os.path.join(os.path.dirname(file_path), result.group(1))
                    if not os.path.isfile(imported_file):
                        found_errors = True
                        print(f"{file_path} erroneously tried to import {imported_file}")
    print()

    print('Checking all "_target_"s and "cls"s')
    target_regex = re.compile(r"_target_.*[\"'](.+?)[\"']")
    full_class_name_regex = re.compile(r"full_class_name.*[\"'](.+?)[\"']")
    for file_path in tqdm(glob.iglob("configs/**/*.json*", recursive=True)):
        with open(file_path, mode="r") as f:
            for line in f.readlines():
                line = line.strip()
                result = target_regex.search(line) or full_class_name_regex.search(line)
                if result is not None:
                    path = result.group(1)
                    if "." in path and " " not in path:
                        try:
                            _locate(path)
                        except ImportError as e:
                            found_errors = True
                            print(f"{file_path} erroneously tried to load {path}")
    print()

    if not found_errors:
        print("Success!")
    else:
        print("Failure. See errors above.")
