import json
import os
import shutil

import yaml


def load_json(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)