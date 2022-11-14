import sys
from pathlib import Path
src_path = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(src_path))

import argparse
import json

from src.utils.eval_utils import get_metrics
from src.utils.load_params import load_params


def evaluate(params):
    data_dir = Path(params.data_load.data_dir)
    test_img_dir_path = Path(params.data_split.test_img_dir_path)
    test_mask_dir_path = Path(params.data_split.test_mask_dir_path)
    test_img_out_dir = data_dir/'test_preds'
    test_img_out_dir.mkdir(exist_ok=True)
    metrics_file_path = params.evaluate.metrics_file
    save_test_preds = params.evaluate.save_test_preds
    img_size = params.train.img_size
    model_pickle_fpath = Path(params.train.model_pickle_fpath).absolute()
    metrics = get_metrics(test_img_dir_path=test_img_dir_path,
                          test_mask_dir_path=test_mask_dir_path,
                          model_pickle_fpath=model_pickle_fpath,
                          test_img_out_dir=test_img_out_dir,
                          img_size=img_size,
                          save_test_preds=save_test_preds)
    Path(params.evaluate.metrics_file).parent.mkdir(parents=True, exist_ok=True)
    json.dump(
        obj=metrics,
        fp=open(metrics_file_path, 'w'),
        indent=4
    )


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    params_path = args.config
    params = load_params(params_path)
    evaluate(params)
