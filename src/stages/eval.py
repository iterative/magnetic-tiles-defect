import argparse
import json
from pathlib import Path

from src.eval_utils import get_metrics
from src.load_params import load_params


def evaluate(params_path):
    params = load_params(params_path)
    test_img_dir_path = Path(params.data_split.test_img_dir_path)
    test_mask_dir_path = Path(params.data_split.test_mask_dir_path)
    reports_dir_path = Path(params.base.reports_dir_path)
    reports_dir_path.mkdir(exist_ok=True)
    img_out_dir = reports_dir_path/'test_preds'
    img_out_dir.mkdir(exist_ok=True)
    metrics_file_path = reports_dir_path/params.evaluate.metrics_file
    save_test_preds = params.evaluate.save_test_preds
    img_size = params.train.img_size
    model_pickle_dir_path = Path(params.train.model_pickle_dir_path)
    model_pickle_fname = params.train.model_pickle_fname
    model_pickle_path = (model_pickle_dir_path/model_pickle_fname).absolute()
    metrics = get_metrics(test_img_dir_path=test_img_dir_path,
                          test_mask_dir_path=test_mask_dir_path,
                          model_pickle_path=model_pickle_path,
                          img_out_dir=img_out_dir,
                          img_size=img_size,
                          save_test_preds=save_test_preds)
    json.dump(
        obj=metrics,
        fp=open(metrics_file_path, 'w')
    )


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    evaluate(params_path=args.config)
