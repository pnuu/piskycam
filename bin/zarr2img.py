#!/usr/bin/env python3

import sys
import os

import numpy as np
import zarr

from piskycam.image import save_max, save_ave


def main():
    """Create an image from piskycam ZARR file."""
    fname_in = sys.argv[1]
    fname_out = sys.argv[2]
    with zarr.open(fname_in, "r") as fid:
        parts = os.path.splitext(fname)
        if "max" in fid:
            fname = parts[0] + "_max" + parts[-1]
            save_max(fid, fname)
        if "sum" in fid:
            fname = parts[0] + "_ave" + parts[-1]
            save_ave(fid, fname)


if __name__ == "__main__":
    main()
