from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

from logging_formatter import Logger
import glob
import pickle
import tensorflow as tf
import numpy as np
import shutil
import time

from bls2017_comp_seg import  train, eval
from eval_seg import eval_seg
logger = Logger()


def run_exp(exp_args):
    # create exp dir
    expDir = os.path.join(exp_args.outdir, exp_args.exp_name)
    lambdas = args.lambdas.split(',')
    lambdas = [float(x) for x in lambdas]

    if not os.path.exists(expDir):
        os.makedirs(expDir)
        logger.info('Creating experiment directory ' + expDir + '/')
    else:
        logger.info('Experiment directory already exist')

    for lmbda in lambdas:
        # for each lambda create subdir
        train_dir = os.path.join(expDir, 'lambda_'+str(lmbda))
        train_time_path = os.path.join(train_dir, 'time_analysis.txt')
        exp_args.checkpoint_dir = train_dir
        exp_args.lmbda = lmbda

        if not exp_args.test_only:
            if not os.path.exists(train_dir):
                logger.info('Creating subdir in experiment directory for lambda = '+str(lmbda))
                os.makedirs(train_dir)
                logger.info('Saving a copy of the code used for running this experiment')
                os.makedirs(os.path.join(train_dir,'code'))
                code_files = os.listdir('examples/')

                for code_file in code_files:
                    _, ext = os.path.splitext(code_file)
                    if ext == '.py' or ext =='.ipynb':
                        code_file = os.path.join('examples/', code_file)
                        shutil.copy(code_file, os.path.join(train_dir, 'code'))
            else:
                logger.warn('Trained with lambda= '+str(lmbda)+' before, skipping')
                #continue

            try:
                logger.info('Start training')
                train_time_st = time.time()
                train(exp_args)
                tf.reset_default_graph()

                train_time_secs = int(time.time() - train_time_st)

            except Exception as e:
                logger.error(str(e))
                #shutil.rmtree(train_dir)
                raise

        test_time_st = time.time()

        highest_val_miou, best_chekpnt = eval_seg(checkpoint_dir=train_dir,
                                                 eval_preprocess_threads=exp_args.preprocess_threads,
                                                 eval_repeatedly=False)
        tf.reset_default_graph()                                                 

        eval(exp_args, expDir, lmbda, best_chekpnt, highest_val_miou)
        tf.reset_default_graph()
        test_time_secs = int(time.time() - test_time_st)

        if not exp_args.test_only:
            time_analysis = {'training took (sec)':train_time_secs,'testing took (sec)': test_time_secs }
            f = open(train_time_path, "w")
            for k, v in time_analysis.items():
                f.write(str(k) + ':' + str(v) + '\n')
            f.close()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument(
      "command", choices=["train", "compress", "decompress", "exp"],
      help="What to do: 'train' loads training data and trains (or continues "
           "to train) a new model. 'compress' reads an image file (lossless "
           "PNG format) and writes a compressed binary file. 'decompress' "
           "reads a binary file and reconstructs the image (in PNG format). "
           "input and output filenames need to be provided for the latter "
           "two options.")
  parser.add_argument(
      "input", nargs="?",
      help="Input filename.")
  parser.add_argument(
      "output", nargs="?",
      help="Output filename.")
  parser.add_argument(
      "--verbose", "-v", action="store_true",
      help="Report bitrate and distortion when training or compressing.")
  parser.add_argument(
      "--num_filters", type=int, default=128,
      help="Number of filters per layer.")
  parser.add_argument(
      "--checkpoint_dir", default="train",
      help="Directory where to save/load model checkpoints.")
  parser.add_argument(
      "--batchsize", type=int, default=4,
      help="Batch size for training.")
  parser.add_argument(
      "--patchsize", type=int, default=512,
      help="Size of image patches for training.")
  parser.add_argument(
      "--last_step", type=int, default=2000000,
      help="Train up to this number of steps.")
  parser.add_argument(
      "--preprocess_threads", type=int, default=8,
      help="Number of CPU threads to use for parallel decoding of training "
           "images.")
  # -----------------------
  parser.add_argument(
      "--exp_name", type=str, default='exp',
      help="Name of the exp directory")
  parser.add_argument(
      "--exp_description", type=str, default='',
      help="details of model architecture used, dataset ...")
  parser.add_argument(
      "--lambdas", type=str, default='64,1024',
      help="list of lambda values that the model will be trained with")
  parser.add_argument(
      "--outdir", type=str, default='/datatmp/Experiments/belbarashy/exps/',
      help="")
  parser.add_argument(
      "--gpu", type=str, default='0',
      help="")
  parser.add_argument(
      "--test_only", "-t", action="store_true",
      help="test the trained model")

  args = parser.parse_args()
  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

  if args.command == "train":
    #train(args)
    pass
  elif args.command == "compress":
    if args.input is None or args.output is None:
      raise ValueError("Need input and output filename for compression.")
    #compress(args)
  elif args.command == "decompress":
    if args.input is None or args.output is None:
      raise ValueError("Need input and output filename for decompression.")
    #decompress(args)
  elif args.command == "exp":
    run_exp(args)
