import json
import os
import shutil

import yaml


def load_json(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)


def dump_json(data_dict, json_file, group_num = 1):
    if group_num == 1:
        with open(json_file, 'w') as f:
            json.dump(data_dict, f, indent=2, separators=(',', ': '))

    else:
        json_path = json_file.replace('.json', '')
        if os.path.exists(json_path):
            shutil.rmtree(json_path)
        os.makedirs(json_path)

        data_list = [data_dict[i:i+group_num] for i in range(0, len(data_dict), group_num)]
        for i, data in enumerate(data_list):
            with open(json_path + '/grp_{}.json'.format(i), 'w') as f:
                json.dump(data, f, indent=2, separators=(',', ': '))


def dump_yaml(data, dump_dir):
    with open(dump_dir, "w") as f:
        yaml.dump(data, f, sort_keys=False)
