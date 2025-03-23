import re
from os import PathLike
import yaml

def get_base_id(id: str) -> str:
    match = re.match(r'\b(?:Article|Art\.|Rule|R\.)\s\d+[a-zA-Z]?\b', id)
    return match.group()

def load_config(config_path: PathLike) -> dict:
    with open(config_path, 'r') as ys:
        config = yaml.safe_load(ys)
    return config