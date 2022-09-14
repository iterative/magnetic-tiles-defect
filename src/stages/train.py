import sys
from pathlib import Path
src_path = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(src_path))

import argparse

from src.utils.load_params import load_params
from src.utils.train_utils import train_model


def train_and_save_model(params):
    random_state = params.base.random_state
    train_img_dir_path = Path(params.data_split.train_img_dir_path)
    train_mask_dir_path = Path(params.data_split.train_mask_dir_path)
    img_size = params.train.img_size
    valid_pct = params.train.valid_pct
    learning_rate = params.train.learning_rate
    batch_size = params.train.batch_size
    n_epochs = params.train.n_epochs
    use_cpu = params.train.use_cpu
    augmentations = params.train.augmentations
    model_pickle_fpath = Path(params.train.model_pickle_fpath).absolute()
    train_model(train_img_dir_path=train_img_dir_path,
                train_mask_dir_path=train_mask_dir_path,
                n_epochs=n_epochs,
                lr=learning_rate,
                use_cpu=use_cpu,
                batch_size=batch_size,
                model_pickle_fpath=model_pickle_fpath,
                valid_pct=valid_pct,
                img_size=img_size,
                augmentations=augmentations,
                seed=random_state)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    params_path = args.config
    params = load_params(params_path)
    train_and_save_model(params)
