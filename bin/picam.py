#!/usr/bin/env python3

import sys
import datetime as dt

import yaml

from piskycam import SkyCam, sun_elevation_now


def read_config(fname):
    """"Read configuration file."""
    with open(fname, 'r') as fid:
        config = yaml.load(fid, Loader=yaml.SafeLoader)
    return config


def sun_is_down(lat, lon, elevation_limit):
    """Return True if Sun is below the horizon."""
    elevation = sun_elevation_now(lat, lon)
    if elevation <= elevation_limit:
        return True
    return False


def main():
    """Run SkyCam."""
    config = read_config(sys.argv[1])
    if sun_is_down(config['latitude'], config['longitude'], config.get('sun_elevation_limit', 0.0)):
        skycam = SkyCam(config)
        skycam.run()


if __name__ == "__main__":
    main()
