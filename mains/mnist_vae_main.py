import sys
import os 

sys.path.extend([os.path.join(sys.path[0],'..')])

import tensorflow as tf

from data_loader.mnist_loader import MnistDataLoaderNumpy
from models.mnist_vae_model import MnistVaeModel
from trainers.mnist_vae_trainer import MnistVaeTrainer

from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import DefinedSummarizer
from utils.utils import get_args


def main():
    # capture the config path from the run arguments
    # then process the json configration file
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir, config.etc_dir])

    # create tensorflow session
    sess = tf.Session()

    # create your data generator
    data_loader = MnistDataLoaderNumpy(config)

    # create instance of the model you want
    model = MnistVaeModel(data_loader, config)

    # create tensorboard logger
    logger = DefinedSummarizer(sess, summary_dir=config.summary_dir, 
                               scalar_tags=['train/loss_per_epoch', 'train/acc_per_epoch',
                                            'test/loss_per_epoch','test/acc_per_epoch'])

    # create trainer and path all previous components to it
    trainer = MnistVaeTrainer(sess, model, config, logger, data_loader)

    # here you train your model
    trainer.train()


if __name__ == '__main__':
    main()
