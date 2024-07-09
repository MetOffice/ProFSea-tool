"""
Copyright (c) 2023, Met Office
All rights reserved.
"""

import os
import pathlib
import yaml

path = pathlib.Path(__file__).parents[0].as_posix()
print(path)

with open(os.path.join(path, "user-settings-greg.yml"), "r") as f:
    settings = yaml.load(f, Loader=yaml.SafeLoader)
