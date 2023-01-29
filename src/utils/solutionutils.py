from os import getcwd
from pathlib import Path


def get_project_root() -> str:
    return str(Path(getcwd().split('src')[0] + '/'))