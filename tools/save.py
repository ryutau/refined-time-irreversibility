import os
import sys
from subprocess import check_output


def repo_path(path=""):
    """Returns absolute path from repository relative path
    This function should be called in repository.

    Args:
        path (string): repository relative path
    Returns:
        string: absolute path
    Examples:
        >>> repo_path('src')
        '/home/username/repo/src'
        >>> repo_path()
        '/home/username/repo/'
    """
    repo_root = check_output(
        ["git", "rev-parse", "--show-toplevel"], universal_newlines=True
    ).rstrip()
    return os.path.join(repo_root, path)


def save_dir(target_dir):
    path = repo_path(f"output/{target_dir}")
    os.makedirs(path, exist_ok=True)
    sys.stdout.write(f'File is saved at "{path}".\n')
    return path
