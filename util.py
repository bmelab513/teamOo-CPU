from datetime import datetime
import importlib
import subprocess
import os
import logging
import numpy as np

from quadratic_weighted_kappa import quadratic_weighted_kappa


def float32(k):
    return np.cast['float32'](k)


def kappa(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = y_true.dot(range(y_true.shape[1]))
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = y_pred.dot(range(y_pred.shape[1]))
    try:
        return quadratic_weighted_kappa(y_true, y_pred)
    except IndexError:
        return np.nan


def kappa_from_proba(w, p, y_true):
    return kappa(y_true, p.dot(w))


def load_module(mod):
    return importlib.import_module(mod.replace('/', '.').split('.py')[0])


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError:
        pass


def get_commit_sha():
    p = subprocess.Popen(['git', 'rev-parse', '--short', 'HEAD'],
                         stdout=subprocess.PIPE)
    output, _ = p.communicate()
    return output.strip().decode('utf-8')


def get_submission_filename():
    sha = get_commit_sha()
    return "data/sub_{}_{}.csv".format(sha,
                                       datetime.now().replace(microsecond=0))
def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)
