#!/usr/bin/env python3
import h5py
from pathlib import Path
from tqdm import tqdm
from fire import Fire


def find_broken_h5s_(feature_dir: str) -> None:
    for h5 in tqdm(list(Path(feature_dir).glob("*.h5"))):
        try:
            with h5py.File(h5):
                pass
        except OSError:
            print(f"Error while opening file {h5}")


if __name__ == "__main__":
    Fire(find_broken_h5s_)
