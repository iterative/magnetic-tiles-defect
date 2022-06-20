
import yaml
from box import ConfigBox


def load_params(params_file):
    with open(params_file, "r") as f:
        params = yaml.safe_load(f)
        params = ConfigBox(params)
    return params
