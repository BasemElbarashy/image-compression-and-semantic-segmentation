import six
import os
import tensorflow as tf
from tensorflow.python.ops import math_ops
import cv2
from deeplab import common
from deeplab import model
from deeplab.datasets import data_generator
from deeplab.utils import train_utils
from deeplab.utils import get_dataset_colormap

from seg_exp import seg_decoder, seg_encoder
from logging_formatter import Logger
from bls2017_comp_seg import build_model
import numpy as np
flags = tf.app.flags
FLAGS = flags.FLAGS
logger = Logger()


# When fine_tune_batch_norm=True, use at least batch size larger than 12
# (batch size more than 16 is better). Otherwise, one could use smaller batch
# size and set fine_tune_batch_norm=False.
flags.DEFINE_integer('eval_batch_size', 1,
                     'The number of images in each batch during training.')

flags.DEFINE_integer('eval_preprocess_threads', 8,
                     'The number of images in each batch during training.')

flags.DEFINE_string('eval_logdir', None, 'Where to write the event logs.')

flags.DEFINE_list('eval_crop_size', '1024,2048',
                  'Image crop size [height, width] during testing.')

flags.DEFINE_multi_float('eval_scales', [1.0],
                         'The scales to resize images for evaluation.')
# Dataset settings.

flags.DEFINE_string('eval_split', 'val',
                    'Which split of the dataset to be used for training')


flags.DEFINE_integer('max_number_of_evaluations', 0,
                     'Maximum number of eval iterations. Will loop '
                     'indefinitely upon nonpositive values.')

flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                     'How often (in seconds) to run evaluation.')

def save_output_samples(checkpoint_dir, best_chekpnt, eval_preprocess_threads=8, eval_crop_size=[1024,2048]):
    eval_batch_size = 1
    compressed_reconstructed_dir = os.path.join(checkpoint_dir, 'compressed_reconstructed_images')
    if not os.path.exists(compressed_reconstructed_dir):
        os.makedirs(compressed_reconstructed_dir)
        logger.info('Creating directory  ' + compressed_reconstructed_dir + '/')


    eval_split = 'val'
    num_sample_output = 20

    dataset = data_generator.Dataset(
        dataset_name='cityscapes',
        split_name=eval_split,
        dataset_dir='/datatmp/Experiments/belbarashy/datasets/Cityscapes/tfrecord/',
        batch_size= eval_batch_size,
        crop_size=[int(sz) for sz in eval_crop_size],
        min_resize_value=None,
        max_resize_value=None,
        resize_factor=None,
        model_variant=None,
        num_readers=eval_preprocess_threads,
        is_training=False,
        should_shuffle=False,
        should_repeat=False)

    samples = dataset.get_one_shot_iterator().get_next()
    in_imgs = samples['image'] / 255
    depth   = samples['depth'] / 255
    labels  = samples['label']
    num_classes = dataset.num_of_classes

    # =================================== arch
    _, _, _, _, _, _, _, _, logits, _ = \
        build_model(in_imgs, depth, None, num_classes, mode='testing')
    # ===================================
    predictions = tf.argmax(logits, 3) # batch*H*W*1

    with tf.Session() as sess:
        if best_chekpnt is None:
            latest = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)
            best_chekpnt = latest
        tf.train.Saver().restore(sess, save_path=best_chekpnt)
        for i in range(num_sample_output):
            test_file_name = str(i)
            depth_path    = os.path.join(compressed_reconstructed_dir, test_file_name + '_depth' + '.png')
            orig_path     = os.path.join(compressed_reconstructed_dir, test_file_name + '_orig' + '.png')
            map_gt_path   = os.path.join(compressed_reconstructed_dir, test_file_name + '_map_gt' + '.png')
            map_pred_path = os.path.join(compressed_reconstructed_dir, test_file_name + '_map_pred' + '.png')


            p, l, input_img, dep = sess.run([predictions, labels, in_imgs, depth])
            l = np.squeeze(l)
            p = np.squeeze(p)
            input_img = np.squeeze(input_img)
            dep       = np.squeeze(dep)
            p[l == 255] = 255

            colored_label = get_dataset_colormap.label_to_color_image(l, 'cityscapes')
            colored_pred  = get_dataset_colormap.label_to_color_image(p, 'cityscapes')

            dep_jet = cv2.applyColorMap(np.uint8(dep * (255 * 2)), cv2.COLORMAP_JET)
            cv2.imwrite(depth_path, dep_jet)
            colored_pred = np.uint8(colored_pred[:, :, ::-1])
            cv2.imwrite(map_pred_path, colored_pred)
            colored_label = np.uint8(colored_label[:, :, ::-1])
            cv2.imwrite(map_gt_path, colored_label)
            input_img = np.uint8(input_img[:, :, ::-1]*255)
            cv2.imwrite(orig_path, input_img)







def eval_seg(checkpoint_dir, eval_preprocess_threads=8, eval_crop_size=[1024,2048],
             eval_logdir='tmp_eval_log/', eval_batch_size=1, eval_repeatedly=False):
    eval_split = 'val'
    dataset = data_generator.Dataset(
        dataset_name='cityscapes',
        split_name=eval_split,
        dataset_dir='/datatmp/Experiments/belbarashy/datasets/Cityscapes/tfrecord/',
        batch_size= eval_batch_size,
        crop_size=[int(sz) for sz in eval_crop_size],
        min_resize_value=None,
        max_resize_value=None,
        resize_factor=None,
        model_variant=None,
        num_readers=eval_preprocess_threads,
        is_training=False,
        should_shuffle=False,
        should_repeat=False)

    tf.gfile.MakeDirs(eval_logdir)
    logger.info('Evaluating on '+eval_split+' set')

    with tf.Graph().as_default():
        samples = dataset.get_one_shot_iterator().get_next()
        # Set shape in order for tf.contrib.tfprof.model_analyzer to work properly.
        samples['image'].set_shape(
            [eval_batch_size,
             int(eval_crop_size[0]),
             int(eval_crop_size[1]),
             3])

        num_classes = dataset.num_of_classes
        in_imgs = samples['image'] / 255
        depth   = samples['depth'] / 255
        labels  = samples['label']
        # =================================== arch
        _, _, _, _, _, _, _, _, logits, _ = \
            build_model(in_imgs, depth, None, num_classes, mode='testing')
        if logits is None:
            highest_val_miou = 0
            best_chekpnt = None
            return highest_val_miou, best_chekpnt
        # ===================================

        predictions = tf.argmax(logits, 3)
        predictions = tf.reshape(predictions, shape=[-1])
        labels = tf.reshape(labels, shape=[-1])
        weights = tf.to_float(tf.not_equal(labels, dataset.ignore_label))
        labels = tf.where(
            tf.equal(labels, dataset.ignore_label), tf.zeros_like(labels), labels)

        predictions_tag = 'miou'
        eval_scales     = [1.0]
        for eval_scale in eval_scales:
          predictions_tag += '_' + str(eval_scale)

        # Define the evaluation metric ==> mIOU over class
        miou, update_op = tf.metrics.mean_iou(predictions, labels, num_classes, weights=weights)
        tf.summary.scalar(predictions_tag, miou)

        summary_op = tf.summary.merge_all()
        summary_hook = tf.contrib.training.SummaryAtEndHook(
            log_dir=eval_logdir, summary_op=summary_op)
        hooks = [summary_hook]

        num_eval_iters = 100000

        tf.contrib.tfprof.model_analyzer.print_model_analysis(
            tf.get_default_graph(),
            tfprof_options=tf.contrib.tfprof.model_analyzer.
            TRAINABLE_VARS_PARAMS_STAT_OPTIONS)

        tf.contrib.tfprof.model_analyzer.print_model_analysis(
            tf.get_default_graph(),
            tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)


        latest = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)
        if eval_repeatedly:
            logger.info('start evaluation repeatedly')
            tf.contrib.training.evaluate_repeatedly(
                master='',
                checkpoint_dir=checkpoint_dir,
                eval_ops=[update_op],
                max_number_of_evaluations=num_eval_iters,
                hooks=hooks,
                eval_interval_secs=eval_interval_secs)
        else:
            logger.info('start evaluating last 5 checkpoints')
            checkpnts_paths = tf.train.get_checkpoint_state(checkpoint_dir=checkpoint_dir).all_model_checkpoint_paths
            best_chekpnt = latest
            highest_val_miou = 0
            for chekpnt_path in checkpnts_paths:
                final_m = tf.contrib.training.evaluate_once(
                    checkpoint_path=chekpnt_path,
                    master='',
                    eval_ops=[update_op],
                    final_ops=miou,
                    hooks=hooks
                )
                if final_m > highest_val_miou:
                    highest_val_miou = final_m
                    best_chekpnt = chekpnt_path
                logger.info(chekpnt_path+' ==> mIOU '+str(final_m))

            logger.info('==============================================')
            logger.info('highest_val_miou = '+str(highest_val_miou))
            logger.info('best_chekpnt = '+str(best_chekpnt))
            logger.info('==============================================')

    if not(eval_repeatedly):
        tf.reset_default_graph()
        save_output_samples(checkpoint_dir, best_chekpnt, eval_preprocess_threads, eval_crop_size)
        return highest_val_miou, best_chekpnt

def main(unused_argv):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    tf.logging.set_verbosity(tf.logging.INFO)

    eval_seg(checkpoint_dir= FLAGS.checkpoint_dir,
             eval_preprocess_threads= FLAGS.eval_preprocess_threads,
             eval_crop_size=FLAGS.eval_crop_size,
             eval_logdir= FLAGS.eval_logdir,
             eval_batch_size= FLAGS.eval_batch_size,
             eval_repeatedly= False)


if __name__ == '__main__':
    flags.mark_flag_as_required('checkpoint_dir')
    flags.mark_flag_as_required('eval_logdir')
    tf.app.run()
