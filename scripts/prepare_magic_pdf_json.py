# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# From: https://raw.githubusercontent.com/opendatalab/MinerU/master/scripts/download_models_hf.py

import json
import os
from urllib.request import urlopen

from download_models import download_mineru_models


def download_json(url):
    with urlopen(url) as response:
        data = json.loads(response.read().decode())
    return data


def download_and_modify_json(url, local_filename, modifications):
    if os.path.exists(local_filename):
        data = json.load(open(local_filename))
        config_version = data.get('config_version', '0.0.0')
        if config_version < '1.2.0':
            data = download_json(url)
    else:
        data = download_json(url)

    for key, value in modifications.items():
        data[key] = value

    with open(local_filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def prepare():
    dirs = download_mineru_models()
    model_dir = dirs['models-dir']
    layoutreader_model_dir = dirs['layoutreader-model-dir']

    print(f'model_dir is: {model_dir}')
    print(f'layoutreader_model_dir is: {layoutreader_model_dir}')

    json_url = 'https://github.com/opendatalab/MinerU/raw/master/magic-pdf.template.json'
    config_file_name = 'magic-pdf.json'
    config_file = os.path.join(".", config_file_name)

    json_mods = {
        'models-dir': model_dir,
        'layoutreader-model-dir': layoutreader_model_dir,
    }

    download_and_modify_json(json_url, config_file, json_mods)
    print(f'The configuration file has been configured successfully, the path is: {config_file}')


if __name__ == '__main__':
    prepare()
