# -*- coding: utf-8 -*-
# Copyright 2018 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicabrle law or agreed to in writing, software
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
import sys
sys.path.append('/home/belbarashy/compression-1.1/')
import tensorflow_compression as tfc
import os
from logging_formatter import Logger
import pickle
import cv2
from deeplab.datasets import data_generator
from deeplab.utils import get_dataset_colormap
from seg_exp import softmax_cross_entropy_loss_mining, CBR, seg_encoder, seg_decoder, fuse_SSMA
from seg_exp import fuse_SSMA_like_sum, fuse_SSMA_like_concat
logger = Logger()

os.environ["CUDA_VISIBLE_DEVICES"]="0"


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


def log_all_summaries(in_imgs, x_tilde, seg_logits, seg_labels, loss, train_bpp, train_mse, seg_loss):
    tf.summary.image('input_image', in_imgs) # tf.summary.image("input_rgb", quantize_image(in_imgs))
    tf.summary.scalar('loss', loss)
    if train_bpp is not None:
        tf.summary.scalar("bpp", train_bpp)
    if train_mse is not None:
        tf.summary.scalar("mse", train_mse * (255 ** 2))
    if seg_loss is not None:
        tf.summary.scalar("seg_cross_entropy", seg_loss)
    if x_tilde is not None:
        tf.summary.image("reconstruction", quantize_image(x_tilde))

    if (seg_logits is not None) and (seg_labels is not None):
        cityscapes_label_colormap = get_dataset_colormap.create_cityscapes_label_colormap()
        cmp = tf.convert_to_tensor(cityscapes_label_colormap, tf.int32)  # (256, 3)
        predictions = tf.expand_dims(tf.argmax(seg_logits, 3), -1)
        summary_predictions = tf.gather(params=cmp, indices=predictions[:,:, :,0])
        summary_label = tf.gather(params=cmp, indices=seg_labels[:,:, :,0])
        semantic_map = tf.cast(summary_predictions, tf.uint8)
        seg_gt = tf.cast(summary_label, tf.uint8)

        tf.summary.image("semantic_map", semantic_map)
        tf.summary.image("label", seg_gt)


def analysis_transform(tensor, num_filters):
  """Builds the analysis transform."""
  with tf.variable_scope("analysis"):
    with tf.variable_scope("layer_0"):
      layer = tfc.SignalConv2D(
          num_filters, (9, 9), corr=True, strides_down=4, padding="same_zeros",
          use_bias=True, activation=tfc.GDN(inverse=False))
      tensor = layer(tensor)

    with tf.variable_scope("layer_1"):
      layer = tfc.SignalConv2D(
          num_filters, (5, 5), corr=True, strides_down=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN(inverse=False))
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

def bls_block(scope, x, num_filters, kernel = (3,3), use_bias = True, activation = 'GDN', strides_down= 1, strides_up=1,
              training=True, drop_rate=None, inverse_gdn = False):

    with tf.variable_scope(scope):
        if activation is not None:
            layer = tfc.SignalConv2D( num_filters, kernel, corr=not(inverse_gdn), strides_down=strides_down, padding="same_zeros",
                                    strides_up=strides_up, use_bias=use_bias, activation=tfc.GDN(inverse=inverse_gdn))
        else:
            layer = tfc.SignalConv2D( num_filters, kernel, corr=not(inverse_gdn), strides_down=strides_down, padding="same_zeros",
                                    strides_up=strides_up, use_bias=use_bias, activation=None)

        x = layer(x)
        if drop_rate is not None:
            x = tf.layers.dropout(x, rate=drop_rate, training=training)

    return x

def seg_encoder_gdn(scope, x, training):
    with tf.variable_scope(scope):
        #conv_kernel = [3,3]
        #pool_ksize  = 2
        drop_rate = 0.4

        x = bls_block('bls1', x, 64, strides_down=1, strides_up=1, training=training, drop_rate=None)
        x = bls_block('bls2', x, 64, strides_down=2, strides_up=1, training=training, drop_rate=None)

        x = bls_block('bls3', x, 128, strides_down=1, strides_up=1, training=training, drop_rate=None)
        skip = bls_block('bls4', x, 128, strides_down=2, strides_up=1, training=training, drop_rate=None)

        x = bls_block('bls5', skip, 256, strides_down=1, strides_up=1, training=training, drop_rate=None)
        x = bls_block('bls6', x, 256, strides_down=1, strides_up=1, training=training, drop_rate=None)
        x = bls_block('bls7', x, 256, strides_down=2, strides_up=1, training=training, drop_rate=drop_rate)

        x = bls_block('bls8', x, 512, strides_down=1, strides_up=1, training=training, drop_rate=None)
        x = bls_block('bls9', x, 512, strides_down=1, strides_up=1, training=training, drop_rate=None)
        x = bls_block('bls10', x, 512, strides_down=2, strides_up=1, training=training, drop_rate=drop_rate)

        x = bls_block('bls11', x, 512, strides_down=1, strides_up=1, training=training, drop_rate=None)
        x = bls_block('bls12', x, 512, strides_down=1, strides_up=1, training=training, drop_rate=None)
        x = bls_block('bls13', x, 512, strides_down=1, strides_up=1, training=training, drop_rate=drop_rate)

    return x, skip

def seg_decoder_gdn(scope, x, training, num_classes, skip=None):
    with tf.variable_scope(scope):
        #conv_kernel = [3, 3]
        #upsampling_factor = 2
        drop_rate = 0.4
        x = bls_block('bls1', x, 512, strides_down=1, strides_up=1, training=training, drop_rate=None, inverse_gdn=True)
        x = bls_block('bls2', x, 512, strides_down=1, strides_up=1, training=training, drop_rate=None, inverse_gdn=True)
        x = bls_block('bls3', x, 512, strides_down=1, strides_up=2, training=training, drop_rate=drop_rate, inverse_gdn=True)

        x = bls_block('bls4', x, 512, strides_down=1, strides_up=1, training=training, drop_rate=None, inverse_gdn=True)
        x = bls_block('bls5', x, 512, strides_down=1, strides_up=1, training=training, drop_rate=None, inverse_gdn=True)
        x = bls_block('bls6', x, 512, strides_down=1, strides_up=2, training=training, drop_rate=drop_rate, inverse_gdn=True)

        x = bls_block('bls7', x, 256, strides_down=1, strides_up=1, training=training, drop_rate=None, inverse_gdn=True)
        x = bls_block('bls8', x, 256, strides_down=1, strides_up=1, training=training, drop_rate=None, inverse_gdn=True)
        x = bls_block('bls9', x, 256, strides_down=1, strides_up=1, training=training, drop_rate=drop_rate, inverse_gdn=True)
        if not(skip is None):
            x = tf.concat([x, skip], 3)

        x = bls_block('bls10', x, 128, strides_down=1, strides_up=2, training=training, drop_rate=None, inverse_gdn=True)
        x = bls_block('bls11', x, 128, strides_down=1, strides_up=1, training=training, drop_rate=None, inverse_gdn=True)

        x = bls_block('bls12', x, 64, strides_down=1, strides_up=2, training=training, drop_rate=None, inverse_gdn=True)
        x = bls_block('bls13', x, 64, strides_down=1, strides_up=1, training=training, drop_rate=None, inverse_gdn=True)

        if num_classes > 0:
            with tf.variable_scope('output'):
                x = tf.layers.conv2d(
                inputs=x,
                filters=num_classes,
                kernel_size=(3,3),
                strides = (1,1),
                use_bias = True,
                padding="same")
            x = tf.nn.relu(x)
        else:
            x = bls_block('output_reconst', x, 3, training=training, inverse_gdn=True, activation=None)
    return x


# seg_arch_for_compAndReconsrurction
def build_model(x_rgb, x_depth, lmbda, num_classes, mode = 'training', seg_labels = None, ignore_label = None):
    # Args
    train_loss, bpp, mse, y_tilde_hat, x_tilde_hat, y, string, entropy_bottleneck, seg_logits = [None]*9
    seg_loss = None
    istraining = (mode == 'training')
    num_pixels = tf.to_float(tf.reduce_prod(tf.shape(x_rgb)[:-1]))

    y, skip = seg_encoder('Encoder', x_rgb, training=istraining)
    entropy_bottleneck = tfc.EntropyBottleneck()
    y_tilde_hat, likelihoods = entropy_bottleneck(y, training=istraining)
    skip = None
    # num_classes = -1 ==> reconstruct the rgb
    x_tilde_hat = seg_decoder('Decoder', y_tilde_hat, training=istraining, num_classes=-1, skip=skip)
    if mode == 'testing':
        string = entropy_bottleneck.compress(y)
        string = tf.squeeze(string, axis=0)

    # Loss
    bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)
    mse = tf.reduce_mean(tf.squared_difference(x_rgb, x_tilde_hat))

    if istraining:
        train_loss = lmbda * mse + bpp

    return train_loss, bpp, mse, x_tilde_hat, y_tilde_hat, y, string, entropy_bottleneck, seg_logits, seg_loss



def train(l_args):
  """Trains the model."""
  if l_args.verbose:
    tf.logging.set_verbosity(tf.logging.INFO)

  # Create input data pipeline.

  dataset = data_generator.Dataset(
      dataset_name='cityscapes',
      split_name='train',
      dataset_dir='/datatmp/Experiments/belbarashy/datasets/Cityscapes/tfrecord/',
      batch_size=l_args.batchsize,
      crop_size=[int(sz) for sz in [l_args.patchsize,l_args.patchsize]],
      min_resize_value=None,
      max_resize_value=None,
      resize_factor=None,
      min_scale_factor=0.5,
      max_scale_factor=2.,
      scale_factor_step_size=0.25,
      model_variant=None,
      num_readers=l_args.preprocess_threads,
      is_training=True,
      should_shuffle=True,
      should_repeat=True)

  # reading batch: keys of samples ['height', 'width', 'image_name', 'label', 'image']
  num_classes = dataset.num_of_classes
  samples = dataset.get_one_shot_iterator().get_next()

  #num_pixels = l_args.batchsize * l_args.patchsize ** 2
  x      = samples['image'] / 255
  depth  = samples['depth'] / 255
  labels = samples['label']

  # Build autoencoder.
  train_loss, train_bpp, train_mse, x_tilde, _, _, _, entropy_bottleneck, seg_logits, seg_loss = \
      build_model(x, depth, l_args.lmbda, num_classes, mode = 'training', seg_labels = labels,
                  ignore_label = dataset.ignore_label)
  
  # Minimize loss and auxiliary loss, and execute update op.
  step = tf.train.get_or_create_global_step()
  main_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
  main_step = main_optimizer.minimize(train_loss, global_step=step)

  if entropy_bottleneck is not None:
    aux_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    aux_step = aux_optimizer.minimize(entropy_bottleneck.losses[0])
    entropy_bottleneck.visualize() ## Creates summary for the probability mass function (PMF) estimated in the bottleneck
    train_op = tf.group(main_step, aux_step, entropy_bottleneck.updates[0])
  else:
    train_op = tf.group(main_step)

  log_all_summaries(x, x_tilde, seg_logits, labels, train_loss, train_bpp, train_mse, seg_loss)

  hooks = [
      tf.train.StopAtStepHook(last_step=l_args.last_step),
      tf.train.NanTensorHook(train_loss),
  ]
  with tf.train.MonitoredTrainingSession(
      hooks=hooks, checkpoint_dir=l_args.checkpoint_dir,
      save_checkpoint_secs=300, save_summaries_secs=60) as sess:
    while not sess.should_stop():
      sess.run(train_op)



def eval(l_args, expDir, lmbda, best_chekpnt, val_miou):
    train_dir = l_args.checkpoint_dir
    metrics_path = os.path.join(train_dir, 'metrics_args.pkl')
    l_args.lmbda = lmbda
    compressed_reconstructed_dir = os.path.join(train_dir, 'compressed_reconstructed_images')
    if not os.path.exists(compressed_reconstructed_dir):
        os.makedirs(compressed_reconstructed_dir)
    val_split_size = 500

    dataset = data_generator.Dataset(
        dataset_name='cityscapes',
        split_name='val',
        dataset_dir='/datatmp/Experiments/belbarashy/datasets/Cityscapes/tfrecord/',
        batch_size=1, #l_args.batchsize
        crop_size=[int(sz) for sz in [1024,2048]],
        min_resize_value=None,
        max_resize_value=None,
        resize_factor=None,
        model_variant=None,
        num_readers=l_args.preprocess_threads,
        is_training=False,
        should_shuffle=False,
        should_repeat=False)

    # reading batch: keys of samples ['height', 'width', 'image_name', 'label', 'image']
    num_classes = dataset.num_of_classes
    samples = dataset.get_one_shot_iterator().get_next()

    x      = samples['image'] / 255
    depth  = samples['depth'] / 255
    labels = samples['label']
    num_pixels = tf.to_float(tf.reduce_prod(tf.shape(x)[:-1]))

    # ======================== Input image dim should be multiple of 16
    x_shape = tf.shape(x)
    x_shape = tf.ceil(x_shape / 16) * 16
    x = tf.image.resize_images(x, (x_shape[1], x_shape[2]))
    # ========================
    """ build model """
    _, eval_bpp, _, x_hat, y_hat, y, string, _, seg_logits, seg_loss = \
        build_model(x, depth, l_args.lmbda, num_classes, mode = 'testing')

    # Bring both images back to 0..255 range.
    x *= 255
    img_file_name = tf.placeholder(tf.string)
    noReconstuction = False
    if x_hat is None:
        noReconstuction = True
        save_reconstructed_op = None
    else:
        x_hat_to_save = tf.identity(x_hat[0, :, :, :])
        x_hat = tf.clip_by_value(x_hat, 0, 1)
        x_hat = tf.round(x_hat * 255)

        mse = tf.reduce_mean(tf.squared_difference(x, x_hat))
        psnr = tf.squeeze(tf.image.psnr(x_hat, x, 255))
        msssim = tf.squeeze(tf.image.ssim_multiscale(x_hat, x, 255))
        # Write reconstructed image out as a PNG file.
        save_reconstructed_op = save_image(img_file_name, x_hat_to_save)





    logger.info('Testing the model on ' + str(val_split_size) + ' images and save the reconstructed images')
    msel, psnrl, msssiml, msssim_dbl, eval_bppl, bppl = [], [], [], [], [], []

    with tf.Session() as sess:
        # Load the latest model checkpoint, get the compressed string and the tensor
        # shapes.
        if best_chekpnt is None:
            latest = tf.train.latest_checkpoint(checkpoint_dir=l_args.checkpoint_dir)
            best_chekpnt = latest
        tf.train.Saver().restore(sess, save_path=best_chekpnt)

        for i in range(val_split_size):
            test_file_name       = str(i)
            compressed_im_path   = os.path.join(compressed_reconstructed_dir,test_file_name+'_compressed'+'.bin')
            reconstucted_im_path = os.path.join(compressed_reconstructed_dir,test_file_name+'_reconstructed'+'.png')
            im_metrics_path      = os.path.join(compressed_reconstructed_dir,test_file_name+'_metrics'+'.pkl')
            l_args.output        = reconstucted_im_path

            if (i<50) and not(noReconstuction):
                eval_bpp_, mse_, psnr_, msssim_, num_pixels_, string_, x_shape, y_shape, _ = \
                    sess.run( [eval_bpp, mse, psnr, msssim, num_pixels, string,
                                tf.shape(x), tf.shape(y), save_reconstructed_op],
                                feed_dict={img_file_name:reconstucted_im_path})
            else:
                if eval_bpp is not None:
                    if noReconstuction:
                        eval_bpp_, num_pixels_, string_, x_shape, y_shape = \
                            sess.run( [eval_bpp, num_pixels, string, tf.shape(x), tf.shape(y)],
                                        feed_dict={img_file_name:reconstucted_im_path})
                        mse_    = 0
                        psnr_   = 0
                        msssim_ = 0
                    else:
                        eval_bpp_, mse_, psnr_, msssim_, num_pixels_, string_, x_shape, y_shape = \
                            sess.run( [eval_bpp, mse, psnr, msssim, num_pixels, string,tf.shape(x), tf.shape(y)],
                                        feed_dict={img_file_name:reconstucted_im_path})
                else:
                    mse_ = 0
                    psnr_ = 0
                    msssim_ = 0
                    eval_bpp_ = 0
                    num_pixels_ = None
                    string_ = None
                    x_shape = None
                    y_shape = None

            if i<50 and (string_ is not None): # save only the first 50 test samples
                # Write a binary file with the shape information and the compressed string.
                with open(compressed_im_path, "wb") as f:
                  f.write(np.array(x_shape[1:-1], dtype=np.uint16).tobytes())
                  f.write(np.array(y_shape[1:-1], dtype=np.uint16).tobytes())
                  f.write(string_)

            if string_ is not None:
                # The actual bits per pixel including overhead.
                bpp_ = (8 + len(string_)) * 8 / num_pixels_
            else:
                bpp_ = 0

            print("Mean squared error: {:0.4f}".format(mse_))
            print("PSNR (dB): {:0.2f}".format(psnr_))
            print("Multiscale SSIM: {:0.4f}".format(msssim_))
            print("Multiscale SSIM (dB): {:0.2f}".format(-10 * np.log10(1 - msssim_)))
            print("Information content in bpp: {:0.4f}".format(eval_bpp_))
            print("Actual bits per pixel: {:0.4f}".format(bpp_))
            msssim_db_ = (-10 * np.log10(1 - msssim_))

            im_metrics = {'mse': mse_, 'psnr': psnr_, 'msssim': msssim_, 'msssim_db': msssim_db_, 'eval_bpp': eval_bpp_,
                          'bpp': bpp_}
            with open(im_metrics_path, "wb") as fp:
                pickle.dump(im_metrics, fp)

            msel.append(mse_)
            psnrl.append(psnr_)
            msssiml.append(msssim_)
            msssim_dbl.append(msssim_db_)
            eval_bppl.append(eval_bpp_)
            bppl.append(bpp_)

    logger.info('Averaging metrics and save them with the exp_args in pickle file metrics_args.pkl' )
    mse_ = np.mean(msel)
    psnr_ = np.mean(psnrl)
    msssim_ = np.mean(msssiml)
    eval_bpp_ = np.mean(eval_bppl)
    bpp_ = np.mean(bppl)
    msssim_db_ = np.mean(msssim_dbl)

    logger.info('MSE        = ' + str(mse_))
    logger.info('PSNR       = ' + str(psnr_))
    logger.info('MS-SSIM    = ' + str(msssim_))
    logger.info('MS-SSIM db = ' + str(msssim_db_))
    logger.info('Eval_bpp   = ' + str(eval_bpp_))
    logger.info('bpp        = ' + str(bpp_))
    logger.info('mIOU       = ' + str(val_miou))
    exp_avg_metrics = {'mse': mse_, 'psnr': psnr_, 'msssim': msssim_, 'msssim_db': msssim_db_, 'eval_bpp': eval_bpp_,
                       'bpp': bpp_,'mIOU': val_miou, 'chk_pnt':best_chekpnt}

    with open(metrics_path, "wb") as fp:
        pickle.dump({'exp_avg_metrics': exp_avg_metrics, 'exp_args': l_args}, fp)


