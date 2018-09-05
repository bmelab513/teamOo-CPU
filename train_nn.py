"""Conv Nets training script."""
import click
import numpy as np
np.random.seed(9)
import time
import data
import util
from nn import create_net
from util import set_logger

@click.command()
@click.option('--cnf', default='configs/c_512_4x4_32.py', show_default=True,
              help='Path or name of configuration module.')
@click.option('--weights_from', default=None, show_default=True,
              help='Path to initial weights file.')
def main(cnf, weights_from):

    config = util.load_module(cnf).config

    if weights_from is None:
        weights_from = config.weights_file
    else:
        weights_from = str(weights_from)

    files = data.get_image_files(config.get('train_dir'))
    names = data.get_names(files)
    labels = data.get_labels(names).astype(np.float32)

    net = create_net(config)

    try:
        net.load_params_from(weights_from)
        print("loaded weights from {}".format(weights_from))
    except IOError:
        print("couldn't load weights starting from scratch")
    print("Shape of files: " + str(files.shape))
    print("Shape of labels: " + str(labels.shape))
    start = time.time()
    print("fitting ...")
    net.fit(files, labels)
    end = time.time()
    print("Time elapsed for fitting: " + str(end-start))
if __name__ == '__main__':
    set_logger('train.log')
    main()

