from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

from logging_formatter import Logger
import glob
import pickle
import tensorflow as tf
from bls2017 import  train, compress, decompress, depth_train, depth_compress
import numpy as np
import shutil
import time

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
        metrics_path = os.path.join(train_dir, 'metrics_args.pkl')
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
                continue

            try:
                logger.info('Start training')
                train_time_st = time.time()

                if exp_args.depth:
                    depth_train(exp_args)
                else:
                    train(exp_args)

                tf.reset_default_graph()

                train_time_secs = int(time.time() - train_time_st)

            except Exception as e:
                logger.error(str(e))
                shutil.rmtree(train_dir)
                raise


        compressed_reconstructed_dir = os.path.join(train_dir, 'compressed_reconstructed_images')
        os.makedirs(compressed_reconstructed_dir)

        test_files= glob.glob(exp_args.test_glob)
        test_files.sort()
        test_files = test_files[ 0:min(exp_args.maxtestimgs, len(test_files))]

        if exp_args.depth:
            test_depth_files= glob.glob(exp_args.test_depth_glob)
            test_depth_files.sort()
            test_depth_files = test_depth_files[ 0:min(exp_args.maxtestimgs, len(test_depth_files))]

        logger.info('Testing the model on '+str(len(test_files))+' images and save the reconstructed images')

        msel, psnrl, msssiml, msssim_dbl, eval_bppl, bppl = [], [], [], [], [], []
        test_time_st = time.time()

        for i, test_file in enumerate(test_files):

            test_file_name       = os.path.splitext(os.path.split(test_file)[1])[0]
            compressed_im_path   = os.path.join(compressed_reconstructed_dir,test_file_name+'_compressed'+'.bin')
            reconstucted_im_path = os.path.join(compressed_reconstructed_dir,test_file_name+'_reconstructed'+'.png')
            im_metrics_path      = os.path.join(compressed_reconstructed_dir,test_file_name+'_metrics'+'.pkl')

            if exp_args.depth:
                exp_args.input  = (test_file, test_depth_files[i])
                exp_args.output = compressed_im_path
                mse, psnr, msssim, msssim_db, eval_bpp, bpp = depth_compress(exp_args)
            else:
                exp_args.input  = test_file
                exp_args.output = compressed_im_path
                mse, psnr, msssim, msssim_db, eval_bpp, bpp = compress(exp_args)


            im_metrics = {'mse':mse,'psnr':psnr, 'msssim':msssim,'msssim_db':msssim_db,'eval_bpp':eval_bpp,'bpp':bpp}
            with open(im_metrics_path, "wb") as fp:
                pickle.dump(im_metrics, fp)
            msel.append(mse)
            psnrl.append(psnr)
            msssiml.append(msssim)
            msssim_dbl.append(msssim_db)
            eval_bppl.append(eval_bpp)
            bppl.append(bpp)

            tf.reset_default_graph()
            exp_args.input  = compressed_im_path
            exp_args.output = reconstucted_im_path
            decompress(exp_args)
            tf.reset_default_graph()
        test_time_secs = int(time.time() - test_time_st)

        logger.info('Averaging metrics and save them with the exp_args in pickle file metrics_args.pkl' )

        mse = np.mean(msel)
        psnr = np.mean(psnrl)
        msssim = np.mean(msssiml)
        eval_bpp = np.mean(eval_bppl)
        bpp = np.mean(bppl)
        msssim_db = np.mean(msssim_dbl)

        logger.info('MSE        = '+str(mse))
        logger.info('PSNR       = '+str(psnr))
        logger.info('MS-SSIM    = '+str(msssim))
        logger.info('MS-SSIM db = '+str(msssim_db))
        logger.info('Eval_bpp   = '+str(eval_bpp))
        logger.info('bpp        = '+str(bpp))
        exp_avg_metrics = {'mse': mse, 'psnr': psnr, 'msssim': msssim,'msssim_db':msssim_db, 'eval_bpp': eval_bpp, 'bpp': bpp}

        with open(metrics_path, "wb") as fp:
            pickle.dump({'exp_avg_metrics': exp_avg_metrics, 'exp_args': exp_args}, fp)
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
      "--train_glob", default="images/*.png",
      help="Glob pattern identifying training data. This pattern must expand "
           "to a list of RGB images in PNG format.")
  parser.add_argument(
      "--batchsize", type=int, default=8,
      help="Batch size for training.")
  parser.add_argument(
      "--patchsize", type=int, default=256,
      help="Size of image patches for training.")
  parser.add_argument(
      "--lambda", type=float, default=0.01, dest="lmbda",
      help="Lambda for rate-distortion tradeoff.")
  parser.add_argument(
      "--last_step", type=int, default=1000000,
      help="Train up to this number of steps.")
  parser.add_argument(
      "--preprocess_threads", type=int, default=16,
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
      "--test_glob", default='test_image/*.png',
      help="Glob pattern identifying test data. This pattern must expand" )
  parser.add_argument(
      "--outdir", type=str, default='experiments/',
      help="")
  parser.add_argument(
      "--gpu", type=str, default='0',
      help="")

  parser.add_argument(
      "--maxtrainimgs", type=int, default=100000000,
      help="")

  parser.add_argument(
      "--maxtestimgs", type=int, default=100000000,
      help="")
  parser.add_argument(
      "--test_depth_glob", default='',
      help="Glob pattern identifying test depth data. This pattern must expand" )
  parser.add_argument(
      "--train_depth_glob", default='',
      help="Glob pattern identifying training depth data. This pattern must expand" )
  parser.add_argument(
      "--depth", "-d", action="store_true",
      help="use depth data")
  parser.add_argument(
      "--test_only", "-t", action="store_true",
      help="test the trained model")

  args = parser.parse_args()
  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

  if args.command == "train":
    train(args)
  elif args.command == "compress":
    if args.input is None or args.output is None:
      raise ValueError("Need input and output filename for compression.")
    compress(args)
  elif args.command == "decompress":
    if args.input is None or args.output is None:
      raise ValueError("Need input and output filename for decompression.")
    decompress(args)
  elif args.command == "exp":
    run_exp(args)
