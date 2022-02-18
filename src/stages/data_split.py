import argparse
from pathlib import Path

from src.data_utils import create_test_dataset
from src.load_params import load_params


def data_split(params_path):
    params = load_params(params_path)
    random_state = params.base.random_state
    data_dir = Path(params.data_load.data_dir)
    new_dirname = params.data_load.new_dirname
    img_dir_path = data_dir/new_dirname/'images'
    mask_dir_path = data_dir/new_dirname/'masks'

    train_img_dir_path = Path(params.data_split.train_img_dir_path)
    train_mask_dir_path = Path(params.data_split.train_mask_dir_path)
    test_img_dir_path = Path(params.data_split.test_img_dir_path)
    test_mask_dir_path = Path(params.data_split.test_mask_dir_path)
    test_pct = params.data_split.test_pct

    create_test_dataset(img_dir_path,
                        mask_dir_path,
                        train_img_dir_path,
                        train_mask_dir_path,
                        test_img_dir_path,
                        test_mask_dir_path,
                        test_pct,
                        random_state)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    data_split(params_path=args.config)
