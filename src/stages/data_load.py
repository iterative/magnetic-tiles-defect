import sys
from pathlib import Path
src_path = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(src_path))

import argparse

from src.utils.data_utils import dataset_prep
from src.utils.load_params import load_params


def data_load(params):
    dataset_url = params.data_load.dataset_url
    data_dir = Path(params.data_load.data_dir)
    data_dir.mkdir(exist_ok=True)
    orig_dirname = params.data_load.orig_dirname
    new_dirname = params.data_load.new_dirname
    dataset_prep(dataset_url=dataset_url,
                 data_dir=data_dir,
                 orig_dirname=orig_dirname,
                 new_dirname=new_dirname)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    params_path = args.config
    params = load_params(params_path)
    data_load(params)
