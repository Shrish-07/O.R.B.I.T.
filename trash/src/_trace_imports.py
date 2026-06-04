# moved to trash: import tracing helper

import builtins
import pandas as pd


def main():
    _real_import = builtins.__import__

    def traced_import(name, globals=None, locals=None, fromlist=(), level=0):
        print("IMPORT:", name)
        return _real_import(name, globals, locals, fromlist, level)

    # Install tracer
    builtins.__import__ = traced_import

    try:
        df = pd.read_parquet("data/processed/modeling_dataset_with_target.parquet")
        print("DONE")
    finally:
        # Always restore original import
        builtins.__import__ = _real_import


if __name__ == "__main__":
    main()
