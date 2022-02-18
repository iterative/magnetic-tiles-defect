import argparse

from src.load_params import load_params
from src.python_env_utils import write_pkg_list_to_file


def check_pkgs(params_path):
    params = load_params(params_path)
    pkg_list_fname = params.base.pkg_list_fname
    write_pkg_list_to_file(pkg_list_fname)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    check_pkgs(params_path=args.config)
