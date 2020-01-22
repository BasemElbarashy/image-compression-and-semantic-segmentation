# -*- coding: utf-8 -*-
# Copyright 2018 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Basic nonlinear transform coder for RGB images.

This is a close approximation of the image compression model of
Ballé, Laparra, Simoncelli (2017):
End-to-end optimized image compression
https://arxiv.org/abs/1611.01704

With patches from Victor Xing <victor.t.xing@gmail.com>
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob

# Dependency imports

import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc
import os
from logging_formatter import Logger
import pickle
from seg_exp import fuse_SSMA, fuse_SSMA_noBatch
logger = Logger()

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def load_image(filename):
  """Loads a PNG image file."""
  string = tf.read_file(filename)
  image = tf.image.decode_image(string, channels=3)
  image = tf.cast(image, tf.float32)
  image /= 255

  return image


def load_image_rgbd(filename):
  """Loads a PNG image file."""
  string = tf.read_file(filename[0])
  image = tf.image.decode_image(string, channels=3)
  image = tf.cast(image, tf.float32)
  image /= 255

  string_ = tf.read_file(filename[1])
  image_d = tf.image.decode_image(string_, channels=3)
  image_d = image_d[:, :, 2:3] # synthia
  image_d = tf.cast(image_d, tf.float32)
  #image_d = image_d * 10
  image_d /= 255

  rgbd = tf.concat([image, image_d], 2)

  return rgbd


def quantize_image(image):
  image = tf.clip_by_value(image, 0, 1)
  image = tf.round(image * 255)
  image = tf.cast(image, tf.uint8)
  return image


def save_image(filename, image):
  """Saves an image to a PNG file."""
  image = quantize_image(image)
  string = tf.image.encode_png(image)
  return tf.write_file(filename, string)



def depth_analysis_transform_1(rgb_tensor, depth_tensor, num_filters):
  """Builds the analysis transform."""
  with tf.variable_scope("analysis"):

    # --------------------------------------- rgb branch
    with tf.variable_scope("layer_0"):
      layer = tfc.SignalConv2D(
          num_filters, (9, 9), corr=True, strides_down=4, padding="same_zeros",
          use_bias=True, activation=tf.nn.relu)
      rgb_tensor = layer(rgb_tensor)

    # --------------------------------------- depth branch
    with tf.variable_scope("layer_d0"):
      layer = tfc.SignalConv2D(
          num_filters, (9, 9), corr=True, strides_down=4, padding="same_zeros",
          use_bias=True, activation=tf.nn.relu)
      depth_tensor = layer(depth_tensor)
    # --------------------------------------- fusion

    tf.summary.histogram('rgb_tensor', rgb_tensor)
    tf.summary.histogram('depth_tensor', depth_tensor)
    tensor = rgb_tensor + depth_tensor

    with tf.variable_scope("layer_1"):
      layer = tfc.SignalConv2D(
          num_filters, (5, 5), corr=True, strides_down=2, padding="same_zeros",
          use_bias=True, activation=tf.nn.relu)
      tensor = layer(tensor)

    with tf.variable_scope("layer_2"):
      layer = tfc.SignalConv2D(
          num_filters, (5, 5), corr=True, strides_down=2, padding="same_zeros",
          use_bias=False, activation=None)
      tensor = layer(tensor)

    return tensor


def depth_analysis_transform_2(rgb_tensor, depth_tensor, num_filters):
  """Builds the analysis transform."""
  with tf.variable_scope("analysis"):
    # --------------------------------------- fusion
    tensor = tf.concat([rgb_tensor, depth_tensor], 3)

    with tf.variable_scope("layer_0"):
      layer = tfc.SignalConv2D(
          num_filters, (9, 9), corr=True, strides_down=4, padding="same_zeros",
          use_bias=True, activation=tf.nn.relu)
      tensor = layer(tensor)

    with tf.variable_scope("layer_1"):
      layer = tfc.SignalConv2D(
          num_filters, (5, 5), corr=True, strides_down=2, padding="same_zeros",
          use_bias=True, activation=tf.nn.relu)
      tensor = layer(tensor)

    with tf.variable_scope("layer_2"):
      layer = tfc.SignalConv2D(
          num_filters, (5, 5), corr=True, strides_down=2, padding="same_zeros",
          use_bias=False, activation=None)
      tensor = layer(tensor)

    return tensor


def depth_analysis_transform_3(rgb_tensor, depth_tensor, num_filters):
  """Builds the analysis transform."""
  with tf.variable_scope("analysis"):

    # --------------------------------------- rgb branch
    with tf.variable_scope("layer_0"):
      layer = tfc.SignalConv2D(
          num_filters, (9, 9), corr=True, strides_down=4, padding="same_zeros",
          use_bias=True, activation=tf.nn.relu)
      rgb_tensor = layer(rgb_tensor)

    # --------------------------------------- depth branch
    with tf.variable_scope("layer_d0"):
      layer = tfc.SignalConv2D(
          num_filters, (9, 9), corr=True, strides_down=4, padding="same_zeros",
          use_bias=True, activation=tf.nn.relu)
      depth_tensor = layer(depth_tensor)
    # --------------------------------------- fusion

    tf.summary.histogram('rgb_tensor', rgb_tensor)
    tf.summary.histogram('depth_tensor', depth_tensor)

    tensor = fuse_SSMA_noBatch('SSMA_fusion', rgb_tensor, depth_tensor, training=True, C=num_filters)
    #tensor = rgb_tensor + depth_tensor
    #tensor = tf.concat([rgb_tensor, depth_tensor], 3)
    with tf.variable_scope("layer_1"):
      layer = tfc.SignalConv2D(
          num_filters, (5, 5), corr=True, strides_down=2, padding="same_zeros",
          use_bias=True, activation=tf.nn.relu)
      tensor = layer(tensor)

    with tf.variable_scope("layer_2"):
      layer = tfc.SignalConv2D(
          num_filters, (5, 5), corr=True, strides_down=2, padding="same_zeros",
          use_bias=False, activation=None)
      tensor = layer(tensor)

    return tensor


def analysis_transform(tensor, num_filters):
  """Builds the analysis transform."""
  with tf.variable_scope("analysis"):
    with tf.variable_scope("layer_0"):
      layer = tfc.SignalConv2D(
          num_filters, (9, 9), corr=True, strides_down=4, padding="same_zeros",
          use_bias=True, activation=tf.nn.relu)
      tensor = layer(tensor)

    with tf.variable_scope("layer_1"):
      layer = tfc.SignalConv2D(
          num_filters, (5, 5), corr=True, strides_down=2, padding="same_zeros",
          use_bias=True, activation=tf.nn.relu)
      tensor = layer(tensor)

    with tf.variable_scope("layer_2"):
      layer = tfc.SignalConv2D(
          num_filters, (5, 5), corr=True, strides_down=2, padding="same_zeros",
          use_bias=False, activation=None)
      tensor = layer(tensor)

    return tensor


def synthesis_transform(tensor, num_filters):
  """Builds the synthesis transform."""

  with tf.variable_scope("synthesis"):
    with tf.variable_scope("layer_0"):
      layer = tfc.SignalConv2D(
          num_filters, (5, 5), corr=False, strides_up=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN(inverse=True))
      tensor = layer(tensor)

    with tf.variable_scope("layer_1"):
      layer = tfc.SignalConv2D(
          num_filters, (5, 5), corr=False, strides_up=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN(inverse=True))
      tensor = layer(tensor)

    with tf.variable_scope("layer_2"):
      layer = tfc.SignalConv2D(
          3, (9, 9), corr=False, strides_up=4, padding="same_zeros",
          use_bias=True, activation=None)
      tensor = layer(tensor)

    return tensor


def train(l_args):
  """Trains the model."""
  if l_args.verbose:
    tf.logging.set_verbosity(tf.logging.INFO)

  # Create input data pipeline.
  with tf.device('/cpu:0'):
    train_files   = glob.glob(l_args.train_glob)
    train_files.sort()
    train_files   = train_files[0:min(l_args.maxtrainimgs, len(train_files))]
    logger.info('Training on '+str(len(train_files))+' images with lambda '+str(l_args.lmbda ) )
    train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
    train_dataset = train_dataset.shuffle(buffer_size=len(train_files)).repeat()
    train_dataset = train_dataset.map(
        load_image, num_parallel_calls=l_args.preprocess_threads)
    train_dataset = train_dataset.map(
        lambda x: tf.random_crop(x, (l_args.patchsize, l_args.patchsize, 3)))
    train_dataset = train_dataset.batch(l_args.batchsize)
    train_dataset = train_dataset.prefetch(32)

  num_pixels = l_args.batchsize * l_args.patchsize ** 2

  # Get training patch from dataset.
  x = train_dataset.make_one_shot_iterator().get_next()

  # Build autoencoder.
  y = analysis_transform(x, l_args.num_filters)
  entropy_bottleneck = tfc.EntropyBottleneck()
  y_tilde, likelihoods = entropy_bottleneck(y, training=True)
  x_tilde = synthesis_transform(y_tilde, l_args.num_filters)

  # Total number of bits divided by number of pixels.
  train_bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)

  # Mean squared error across pixels.
  train_mse = tf.reduce_mean(tf.squared_difference(x, x_tilde))
  # Multiply by 255^2 to correct for rescaling.
  # train_mse *= 255 ** 2

  # The rate-distortion cost.
  train_loss = l_args.lmbda * train_mse + train_bpp

  # Minimize loss and auxiliary loss, and execute update op.
  step = tf.train.create_global_step()
  main_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
  main_step = main_optimizer.minimize(train_loss, global_step=step)

  aux_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
  aux_step = aux_optimizer.minimize(entropy_bottleneck.losses[0])

  train_op = tf.group(main_step, aux_step, entropy_bottleneck.updates[0])

  tf.summary.scalar("loss", train_loss)
  tf.summary.scalar("bpp", train_bpp)
  tf.summary.scalar("mse", train_mse*(255 ** 2))
  tf.summary.image("original", quantize_image(x))
  tf.summary.image("reconstruction", quantize_image(x_tilde))

  # Creates summary for the probability mass function (PMF) estimated in the
  # bottleneck.
  entropy_bottleneck.visualize()

  hooks = [
      tf.train.StopAtStepHook(last_step=l_args.last_step),
      tf.train.NanTensorHook(train_loss),
  ]
  with tf.train.MonitoredTrainingSession(
      hooks=hooks, checkpoint_dir=l_args.checkpoint_dir,
      save_checkpoint_secs=300, save_summaries_secs=60) as sess:
    while not sess.should_stop():
      sess.run(train_op)

def depth_train(l_args):
  """Trains the model."""
  if l_args.verbose:
    tf.logging.set_verbosity(tf.logging.INFO)

  # Create input data pipeline.
  with tf.device('/cpu:0'):
    train_files_rgb   = glob.glob(l_args.train_glob)
    train_files_rgb.sort()
    train_files_depth   = glob.glob(l_args.train_depth_glob)
    train_files_depth.sort()
    train_files = [f for f in zip(train_files_rgb, train_files_depth)]

    train_files   = train_files[0:min(l_args.maxtrainimgs, len(train_files))]
    logger.info('Training on '+str(len(train_files))+' images (RGB+D) with lambda '+str(l_args.lmbda ) )
    train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
    train_dataset = train_dataset.shuffle(buffer_size=len(train_files)).repeat()
    train_dataset = train_dataset.map(
        load_image_rgbd, num_parallel_calls=l_args.preprocess_threads)

    train_dataset = train_dataset.map(
        lambda x: tf.random_crop(x, (l_args.patchsize, l_args.patchsize, 4)))


    train_dataset = train_dataset.batch(l_args.batchsize)
    train_dataset = train_dataset.prefetch(32)

  num_pixels = l_args.batchsize * l_args.patchsize ** 2

  # Get training patch from dataset.
  x_rgbd  = train_dataset.make_one_shot_iterator().get_next()
  x,depth = tf.split(x_rgbd, [3,1], 3)

  # Build autoencoder.
  #y = analysis_transform(x, l_args.num_filters)
  y = depth_analysis_transform_3(x, depth, l_args.num_filters)
  entropy_bottleneck = tfc.EntropyBottleneck()
  y_tilde, likelihoods = entropy_bottleneck(y, training=True)
  x_tilde = synthesis_transform(y_tilde, l_args.num_filters)

  # Total number of bits divided by number of pixels.
  train_bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)

  # Mean squared error across pixels.
  train_mse = tf.reduce_mean(tf.squared_difference(x, x_tilde))
  # Multiply by 255^2 to correct for rescaling.
  # train_mse *= 255 ** 2

  # The rate-distortion cost.
  train_loss = l_args.lmbda * train_mse + train_bpp

  # Minimize loss and auxiliary loss, and execute update op.
  step = tf.train.create_global_step()
  main_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
  main_step = main_optimizer.minimize(train_loss, global_step=step)

  aux_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
  aux_step = aux_optimizer.minimize(entropy_bottleneck.losses[0])

  train_op = tf.group(main_step, aux_step, entropy_bottleneck.updates[0])

  tf.summary.scalar("loss", train_loss)
  tf.summary.scalar("bpp", train_bpp)
  tf.summary.scalar("mse", train_mse*(255 ** 2))
  tf.summary.image("input_rgb", quantize_image(x))
  tf.summary.image("input_depth", quantize_image(depth))
  tf.summary.image("reconstruction", quantize_image(x_tilde))

  # Creates summary for the probability mass function (PMF) estimated in the
  # bottleneck.
  entropy_bottleneck.visualize()

  hooks = [
      tf.train.StopAtStepHook(last_step=l_args.last_step),
      tf.train.NanTensorHook(train_loss),
  ]
  with tf.train.MonitoredTrainingSession(
      hooks=hooks, checkpoint_dir=l_args.checkpoint_dir,
      save_checkpoint_secs=300, save_summaries_secs=60) as sess:
    while not sess.should_stop():
      sess.run(train_op)


def compress(l_args):
  """Compresses an image."""

  # Load input image and add batch dimension.
  x = load_image(l_args.input)
  x = tf.expand_dims(x, 0)
  x.set_shape([1, None, None, 3])
  #======================== Input image dim should be multiple of 16
  x_shape = tf.shape(x)
  x_shape = tf.ceil(x_shape/16)*16
  x = tf.image.resize_images(x, (x_shape[1], x_shape[2]))
  #========================
  # Transform and compress the image, then remove batch dimension.
  y = analysis_transform(x, l_args.num_filters)


  entropy_bottleneck = tfc.EntropyBottleneck()
  string = entropy_bottleneck.compress(y)
  string = tf.squeeze(string, axis=0)

  # Transform the quantized image back (if requested).
  y_hat, likelihoods = entropy_bottleneck(y, training=False)
  x_hat = synthesis_transform(y_hat, l_args.num_filters)

  num_pixels = tf.to_float(tf.reduce_prod(tf.shape(x)[:-1]))

  # Total number of bits divided by number of pixels.
  eval_bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)

  # Bring both images back to 0..255 range.
  x *= 255
  x_hat = tf.clip_by_value(x_hat, 0, 1)
  x_hat = tf.round(x_hat * 255)

  mse = tf.reduce_mean(tf.squared_difference(x, x_hat))
  psnr = tf.squeeze(tf.image.psnr(x_hat, x, 255))
  msssim = tf.squeeze(tf.image.ssim_multiscale(x_hat, x, 255))

  with tf.Session() as sess:
    # Load the latest model checkpoint, get the compressed string and the tensor
    # shapes.
    latest = tf.train.latest_checkpoint(checkpoint_dir=l_args.checkpoint_dir)
    tf.train.Saver().restore(sess, save_path=latest)
    string, x_shape, y_shape = sess.run([string, tf.shape(x), tf.shape(y)])

    # Write a binary file with the shape information and the compressed string.
    with open(l_args.output, "wb") as f:
      f.write(np.array(x_shape[1:-1], dtype=np.uint16).tobytes())
      f.write(np.array(y_shape[1:-1], dtype=np.uint16).tobytes())
      f.write(string)

    # If requested, transform the quantized image back and measure performance.

    eval_bpp, mse, psnr, msssim, num_pixels = sess.run([eval_bpp, mse, psnr, msssim, num_pixels])

    # The actual bits per pixel including overhead.
    bpp = (8 + len(string)) * 8 / num_pixels

    print("Mean squared error: {:0.4f}".format(mse))
    print("PSNR (dB): {:0.2f}".format(psnr))
    print("Multiscale SSIM: {:0.4f}".format(msssim))
    print("Multiscale SSIM (dB): {:0.2f}".format(-10 * np.log10(1 - msssim)))
    print("Information content in bpp: {:0.4f}".format(eval_bpp))
    print("Actual bits per pixel: {:0.4f}".format(bpp))
    msssim_db = (-10 * np.log10(1 - msssim))

  return  mse, psnr, msssim, msssim_db, eval_bpp, bpp


def depth_compress(l_args):
    """Compresses an image."""
    # Load input image and add batch dimension.
    x_rgbd = load_image_rgbd(l_args.input)
    x_rgbd = tf.expand_dims(x_rgbd, 0)
    x_rgbd.set_shape([1, None, None, 4])
    # ======================== Input image dim should be multiple of 16
    x_shape = tf.shape(x_rgbd)
    x_shape = tf.ceil(x_shape / 16) * 16
    x_rgbd = tf.image.resize_images(x_rgbd, (x_shape[1], x_shape[2]))
    # ========================
    # Transform and compress the image, then remove batch dimension.
    x, depth = tf.split(x_rgbd, [3, 1], 3)

    y = depth_analysis_transform_3(x, depth, l_args.num_filters)

    entropy_bottleneck = tfc.EntropyBottleneck()
    string = entropy_bottleneck.compress(y)
    string = tf.squeeze(string, axis=0)

    # Transform the quantized image back (if requested).
    y_hat, likelihoods = entropy_bottleneck(y, training=False)
    x_hat = synthesis_transform(y_hat, l_args.num_filters)

    num_pixels = tf.to_float(tf.reduce_prod(tf.shape(x)[:-1]))

    # Total number of bits divided by number of pixels.
    eval_bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)

    # Bring both images back to 0..255 range.
    x *= 255
    x_hat = tf.clip_by_value(x_hat, 0, 1)
    x_hat = tf.round(x_hat * 255)

    mse = tf.reduce_mean(tf.squared_difference(x, x_hat))
    psnr = tf.squeeze(tf.image.psnr(x_hat, x, 255))
    msssim = tf.squeeze(tf.image.ssim_multiscale(x_hat, x, 255))

    with tf.Session() as sess:
        # Load the latest model checkpoint, get the compressed string and the tensor
        # shapes.
        latest = tf.train.latest_checkpoint(checkpoint_dir=l_args.checkpoint_dir)
        tf.train.Saver().restore(sess, save_path=latest)
        string, x_shape, y_shape = sess.run([string, tf.shape(x), tf.shape(y)])

        # Write a binary file with the shape information and the compressed string.
        with open(l_args.output, "wb") as f:
            f.write(np.array(x_shape[1:-1], dtype=np.uint16).tobytes())
            f.write(np.array(y_shape[1:-1], dtype=np.uint16).tobytes())
            f.write(string)

        # If requested, transform the quantized image back and measure performance.

        eval_bpp, mse, psnr, msssim, num_pixels = sess.run([eval_bpp, mse, psnr, msssim, num_pixels])

        # The actual bits per pixel including overhead.
        bpp = (8 + len(string)) * 8 / num_pixels

        print("Mean squared error: {:0.4f}".format(mse))
        print("PSNR (dB): {:0.2f}".format(psnr))
        print("Multiscale SSIM: {:0.4f}".format(msssim))
        print("Multiscale SSIM (dB): {:0.2f}".format(-10 * np.log10(1 - msssim)))
        print("Information content in bpp: {:0.4f}".format(eval_bpp))
        print("Actual bits per pixel: {:0.4f}".format(bpp))
        msssim_db = (-10 * np.log10(1 - msssim))

    return mse, psnr, msssim, msssim_db, eval_bpp, bpp


def decompress(l_args):
  """Decompresses an image."""

  # Read the shape information and compressed string from the binary file.
  with open(l_args.input, "rb") as f:
    x_shape = np.frombuffer(f.read(4), dtype=np.uint16)
    y_shape = np.frombuffer(f.read(4), dtype=np.uint16)
    string = f.read()

  y_shape = [int(s) for s in y_shape] + [l_args.num_filters]

  # Add a batch dimension, then decompress and transform the image back.
  strings = tf.expand_dims(string, 0)
  entropy_bottleneck = tfc.EntropyBottleneck(dtype=tf.float32)
  y_hat = entropy_bottleneck.decompress(
      strings, y_shape, channels=l_args.num_filters)
  x_hat = synthesis_transform(y_hat, l_args.num_filters)

  # Remove batch dimension, and crop away any extraneous padding on the bottom
  # or right boundaries.
  #x_hat = x_hat[0, :x_shape[0], :x_shape[1], :]
  x_hat = x_hat[0, :int(x_shape[0]), :int(x_shape[1]), :]

  # Write reconstructed image out as a PNG file.
  op = save_image(l_args.output, x_hat)

  # Load the latest model checkpoint, and perform the above actions.
  with tf.Session() as sess:
    latest = tf.train.latest_checkpoint(checkpoint_dir=l_args.checkpoint_dir)
    tf.train.Saver().restore(sess, save_path=latest)
    sess.run(op)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument(
      "command", choices=["train", "compress", "decompress"],
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
      "--preprocess_threads", type=int, default=10,
      help="Number of CPU threads to use for parallel decoding of training "
           "images.")

  args = parser.parse_args()

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
