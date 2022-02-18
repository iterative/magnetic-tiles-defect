import argparse
from pathlib import Path

from src.load_params import load_params
from src.train_utils import train_model


def train_and_save_model(params_path):
    params = load_params(params_path)
    random_state = params.base.random_state
    train_img_dir_path = Path(params.data_split.train_img_dir_path)
    train_mask_dir_path = Path(params.data_split.train_mask_dir_path)
    code_names = params.data_load.code_names
    img_size = params.train.img_size
    valid_pct = params.train.valid_pct
    learning_rate = params.train.learning_rate
    batch_size = params.train.batch_size
    n_epochs = params.train.n_epochs
    use_cpu = params.train.use_cpu
    model_pickle_dir_path = Path(params.train.model_pickle_dir_path)
    model_pickle_dir_path.mkdir(exist_ok=True)
    model_pickle_fname = params.train.model_pickle_fname
    model_pickle_path = (model_pickle_dir_path/model_pickle_fname).absolute()
    train_model(train_img_dir_path=train_img_dir_path,
                train_mask_dir_path=train_mask_dir_path,
                code_names=code_names,
                n_epochs=n_epochs,
                lr=learning_rate,
                use_cpu=use_cpu,
                batch_size=batch_size,
                model_pickle_path=model_pickle_path,
                valid_pct=valid_pct,
                img_size=img_size,
                seed=random_state)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    train_and_save_model(params_path=args.config)
