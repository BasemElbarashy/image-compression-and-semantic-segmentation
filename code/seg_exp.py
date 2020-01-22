import six
import os
import tensorflow as tf
from deeplab.datasets import data_generator
from deeplab.utils import get_dataset_colormap

flags = tf.app.flags
FLAGS = flags.FLAGS


# When fine_tune_batch_norm=True, use at least batch size larger than 12
# (batch size more than 16 is better). Otherwise, one could use smaller batch
# size and set fine_tune_batch_norm=False.

flags.DEFINE_integer('train_batch_size', 4,
                     'The number of images in each batch during training.')

flags.DEFINE_boolean('use_skip', False, '')
flags.DEFINE_boolean('use_skip_1by1', False, '')

flags.DEFINE_list('train_crop_size', '512,512',
                  'Image crop size [height, width] during training.')


flags.DEFINE_float('min_scale_factor', 0.5,
                   'Mininum scale factor for data augmentation.')

flags.DEFINE_float('max_scale_factor', 2.,
                   'Maximum scale factor for data augmentation.')

flags.DEFINE_float('scale_factor_step_size', 0.25,
                   'Scale factor step size for data augmentation.')


# Dataset settings.
flags.DEFINE_string('dataset', 'cityscapes',
                    'Name of the segmentation dataset.')
flags.DEFINE_string('train_split', 'train',
                    'Which split of the dataset to be used for training')
flags.DEFINE_string('dataset_dir', '/datatmp/Experiments/belbarashy/datasets/Cityscapes/tfrecord/',
                    'Where the dataset reside.')

# gpu
flags.DEFINE_string("gpu", '0', 'gpus used for training')

# training
flags.DEFINE_integer("last_step", 50, '')
flags.DEFINE_string("checkpoint_dir", '', '')


#mining
# Hard example mining related flags.
flags.DEFINE_integer(
    'hard_example_mining_step', 0,
    'The training step in which exact hard example mining kicks off. Note we '
    'gradually reduce the mining percent to the specified '
    'top_k_percent_pixels. For example, if hard_example_mining_step=100K and '
    'top_k_percent_pixels=0.25, then mining percent will gradually reduce from '
    '100% to 25% until 100K steps after which we only mine top 25% pixels.')


flags.DEFINE_float(
    'top_k_percent_pixels', 1.0,
    'The top k percent pixels (in terms of the loss values) used to compute '
    'loss during training. This is useful for hard pixel mining.')

"""
with tf.device(tf.train.replica_device_setter(ps_tasks=FLAGS.num_ps_tasks)):
    assert FLAGS.train_batch_size % FLAGS.num_clones == 0, (
        'Training batch size not divisble by number of clones (GPUs).')
    clone_batch_size = FLAGS.train_batch_size // FLAGS.num_clones
"""



def _div_maybe_zero(total_loss, num_present):
    """Normalizes the total loss with the number of present pixels."""
    return tf.to_float(num_present > 0) * tf.div(total_loss,
                                       tf.maximum(1e-5, num_present))


# modified from deeplabV3+ loss with hard mining
def softmax_cross_entropy_loss_mining(logits,
                                      labels,
                                      num_classes,
                                      ignore_label,
                                      loss_weight=1.0,
                                      upsample_logits=True,
                                      hard_example_mining_step=0,
                                      top_k_percent_pixels=1.0,
                                      scope=None):
    """
    Args:
      num_classes: Integer, number of target classes.
      ignore_label: Integer, label to ignore.
      loss_weight: Float, loss weight.
      upsample_logits: Boolean, upsample logits or not.
      hard_example_mining_step: An integer, the training step in which the hard
        exampling mining kicks off. Note that we gradually reduce the mining
        percent to the top_k_percent_pixels. For example, if
        hard_example_mining_step = 100K and top_k_percent_pixels = 0.25, then
        mining percent will gradually reduce from 100% to 25% until 100K steps
        after which we only mine top 25% pixels.
      top_k_percent_pixels: A float, the value lies in [0.0, 1.0]. When its value
        < 1.0, only compute the loss for the top k percent pixels (e.g., the top
        20% pixels). This is useful for hard pixel mining.
      scope: String, the scope for the loss.

    """

    loss_scope = scope

    labels = tf.reshape(labels, shape=[-1])
    scaled_labels = labels

    scaled_labels = tf.reshape(scaled_labels, shape=[-1])
    not_ignore_mask = tf.to_float(tf.not_equal(scaled_labels,
                                               ignore_label)) * loss_weight
    one_hot_labels = tf.one_hot(
        scaled_labels, num_classes, on_value=1.0, off_value=0.0)

    if top_k_percent_pixels == 1.0:
        # Compute the loss for all pixels.
        loss_cl = tf.losses.softmax_cross_entropy(
            one_hot_labels,
            tf.reshape(logits, shape=[-1, num_classes]),
            weights=not_ignore_mask,
            scope=loss_scope)
    else:
        logits = tf.reshape(logits, shape=[-1, num_classes])
        weights = not_ignore_mask
        with tf.name_scope(loss_scope, 'softmax_hard_example_mining',
                           [logits, one_hot_labels, weights]):
            one_hot_labels = tf.stop_gradient(
                one_hot_labels, name='labels_stop_gradient')
            pixel_losses = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=one_hot_labels,
                logits=logits,
                name='pixel_losses')
            weighted_pixel_losses = tf.multiply(pixel_losses, weights)
            num_pixels = tf.to_float(tf.shape(logits)[0])
            # Compute the top_k_percent pixels based on current training step.
            if hard_example_mining_step == 0:
                # Directly focus on the top_k pixels.
                top_k_pixels = tf.to_int32(top_k_percent_pixels * num_pixels)
            else:
                # Gradually reduce the mining percent to top_k_percent_pixels.
                global_step = tf.to_float(tf.train.get_or_create_global_step())
                ratio = tf.minimum(1.0, global_step / hard_example_mining_step)
                top_k_pixels = tf.to_int32(
                    (ratio * top_k_percent_pixels + (1.0 - ratio)) * num_pixels)
            top_k_losses, _ = tf.nn.top_k(weighted_pixel_losses,
                                          k=top_k_pixels,
                                          sorted=True,
                                          name='top_k_percent_pixels')
            total_loss = tf.reduce_sum(top_k_losses)
            num_present = tf.reduce_sum(
                tf.to_float(tf.not_equal(top_k_losses, 0.0)))
            loss_cl = _div_maybe_zero(total_loss, num_present)
    return loss_cl


def normal_loss(logits, labels, num_classes, ignore_label):
    """
    logits, the output from decoder layers, without softmax, shape [Num_batch,height,width,Number_class]
    lables: the atual label information, shape [Num_batch,height,width,1]
    """
    logits = logits + 1e-8

    labels = tf.reshape(labels, shape=[-1])
    one_hot_labels = tf.one_hot(labels, num_classes, on_value=1.0, off_value=0.0)
    loss_weight = 1
    not_ignore_mask = tf.to_float(tf.not_equal(labels,ignore_label)) * loss_weight

    cross_entropy = tf.losses.softmax_cross_entropy(
        one_hot_labels,
        tf.reshape(logits, shape=[-1, num_classes]),
        scope='segmentaion_CI_Loss',
        weights=not_ignore_mask)

    tf.summary.scalar('loss', cross_entropy)

    logits_reshape = tf.reshape(logits, [-1, num_classes])
    #correct_prediction = tf.equal(tf.argmax(logits_reshape, -1), labels)
    #accuracy = tf.reduce_mean(tf.to_float(correct_prediction))
    #tf.summary.scalar('accuracy', accuracy)

    return cross_entropy, None, logits



def resize_bilinear(images, upsampling_factor, output_dtype=tf.float32):
  """Returns resized images as output_type.
  Args:
    images: A tensor of size [batch, height_in, width_in, channels].
    upsampling_factor:
    output_dtype: The destination type.
  Returns:
    A tensor of size [batch, height_out, width_out, channels] as a dtype of
      output_dtype.
  """
  new_shape = tf.shape(images)[1:3] * upsampling_factor
  images    = tf.image.resize_bilinear(images, new_shape, align_corners=True)
  return tf.cast(images, dtype=output_dtype)


def CBR(scope, x, num_filters, kernel, training):
    with tf.variable_scope(scope):
        x = tf.layers.conv2d(
            inputs=x,
            filters=num_filters,
            kernel_size=kernel,
            strides = (1,1),
            use_bias = False,
            padding="same")
        b = tf.Variable(tf.random_normal([num_filters]))
        x = tf.nn.bias_add(x, b)
        x = tf.layers.batch_normalization(x, training= True)
    return tf.nn.relu(x)


def conv_reconstruct_rgb(scope, x, kernel):
    num_filters = 3
    with tf.variable_scope(scope):
        x = tf.layers.conv2d(
            inputs=x,
            filters=num_filters,
            kernel_size=kernel,
            strides = (1,1),
            use_bias = False,
            padding="same")
        b = tf.Variable(tf.random_normal([num_filters]))
        x = tf.nn.bias_add(x, b)
    return x


def max_pooling(scope, x, ksize=2, stride=2):
    with tf.variable_scope(scope):
        return tf.nn.max_pool(x,
                              ksize=[1, ksize, ksize, 1],
                              strides=[1, stride, stride, 1],
                              padding='SAME')


def dropout(scope, x, drop_rate, training):
    with tf.variable_scope(scope):
        return tf.layers.dropout(x, rate=drop_rate, training=training)


# model arch
def seg_encoder(scope, x, training, add_sampling_layer=False):
    with tf.variable_scope(scope):
        conv_kernel = [3,3]
        pool_ksize  = 2
        drop_rate = 0.4  

        x = CBR('CBR1', x, 64, conv_kernel, training)
        x = CBR('CBR2', x, 64, conv_kernel, training)
        x = max_pooling('pool1', x, ksize=pool_ksize, stride=2)

        x = CBR('CBR3', x, 128, conv_kernel, training)
        x = CBR('CBR4', x, 128, conv_kernel, training)
        skip = max_pooling('pool2', x, ksize=pool_ksize, stride=2)

        x = CBR('CBR5', skip, 256, conv_kernel, training)
        x = CBR('CBR6', x, 256, conv_kernel, training)
        x = CBR('CBR7', x, 256, conv_kernel, training)
        x = max_pooling('pool3', x, ksize=pool_ksize, stride=2)
        x = dropout('dropout1', x, drop_rate, training)

        x = CBR('CBR8', x, 512, conv_kernel, training)
        x = CBR('CBR9', x, 512, conv_kernel, training)
        x = CBR('CBR10',x, 512, conv_kernel, training)
        x = max_pooling('pool4', x, ksize=pool_ksize, stride=2)
        x = dropout('dropout2', x, drop_rate, training)

        x = CBR('CBR11', x, 512, conv_kernel, training)
        x = CBR('CBR12', x, 512, conv_kernel, training)
        x = CBR('CBR13', x, 512, conv_kernel, training)
        if add_sampling_layer:
            x = max_pooling('pool5', x, ksize=pool_ksize, stride=2)  #removed to have same latent-input spatial ratio 1/16
        x = dropout('dropout3', x, drop_rate, training)
    return x, skip


def seg_decoder(scope, x, training, num_classes, skip=None, add_sampling_layer=False):
    with tf.variable_scope(scope):
        conv_kernel = [3, 3]
        upsampling_factor = 2
        drop_rate = 0.4
        if add_sampling_layer:
            x = resize_bilinear(x, upsampling_factor)
        x = CBR('CBR1', x, 512, conv_kernel, training)
        x = CBR('CBR2', x, 512, conv_kernel, training)
        x = CBR('CBR3', x, 512, conv_kernel, training)
        x = dropout('dropout1', x, drop_rate, training)

        x = resize_bilinear(x, upsampling_factor )
        x = CBR('CBR4', x, 512, conv_kernel, training)
        x = CBR('CBR5', x, 512, conv_kernel, training)
        x = CBR('CBR6', x, 512, conv_kernel, training)
        x = dropout('dropout2', x, drop_rate, training)

        x = resize_bilinear(x, upsampling_factor )

        x = CBR('CBR7', x, 256, conv_kernel, training)
        x = CBR('CBR8', x, 256, conv_kernel, training)
        x = CBR('CBR9', x, 256, conv_kernel, training)
        if not(skip is None):
            x = tf.concat([x, skip], 3)
        x = dropout('dropout3', x, drop_rate, training)

        x = resize_bilinear(x, upsampling_factor )
        x = CBR('CBR10', x, 128, conv_kernel, training)
        x = CBR('CBR11', x, 128, conv_kernel, training)

        x = resize_bilinear(x, upsampling_factor )
        x = CBR('CBR13', x, 64, conv_kernel, training)
        x = CBR('CBR14', x, 64, conv_kernel, training)
        if num_classes > 0:
            x = CBR('output', x, num_classes, conv_kernel, training)
        else:
            x = conv_reconstruct_rgb('output', x, conv_kernel)
    return x

def fuse_SSMA_noBatch(scope, feats_mod_1, feats_mod_2, training, C=512):
    kernel = [3, 3]
    eta = 16
    C_over_eta = 8

    with tf.variable_scope(scope):
        #feats_shape = tf.shape(feats_mod_1) # assuming both modality feats has same number of channels
        #C = feats_shape[3]
        feats_con = tf.concat([feats_mod_1, feats_mod_2], 3)

        x = tf.layers.conv2d(inputs=feats_con,
            filters=C_over_eta,
            kernel_size=kernel,
            strides = (1,1),
            use_bias = False,
            padding="same")
        b = tf.Variable(tf.random_normal([C_over_eta]))
        x = tf.nn.bias_add(x, b)
        x = tf.nn.relu(x)

        x = tf.layers.conv2d(
            inputs=x,
            filters=2*C,
            kernel_size=kernel,
            strides = (1,1),
            use_bias = False,
            padding="same")
        b = tf.Variable(tf.random_normal([2*C]))
        x = tf.nn.bias_add(x, b)
        gate = tf.nn.sigmoid(x)

        fused_feats = feats_con*gate
        fused_feats = tf.layers.conv2d(
            inputs=fused_feats,
            filters=C,
            kernel_size=kernel,
            strides = (1,1),
            use_bias = False,
            padding="same")
        b = tf.Variable(tf.random_normal([C]))
        fused_feats = tf.nn.bias_add(fused_feats, b)

    return fused_feats


def fuse_SSMA(scope, feats_mod_1, feats_mod_2, training, C=512):
    kernel = [3, 3]
    eta = 16
    C_over_eta = 32

    with tf.variable_scope(scope):
        #feats_shape = tf.shape(feats_mod_1) # assuming both modality feats has same number of channels
        #C = feats_shape[3]
        feats_con = tf.concat([feats_mod_1, feats_mod_2], 3)

        x = tf.layers.conv2d(inputs=feats_con,
            filters=C_over_eta,
            kernel_size=kernel,
            strides = (1,1),
            use_bias = False,
            padding="same")
        b = tf.Variable(tf.random_normal([C_over_eta]))
        x = tf.nn.bias_add(x, b)
        x = tf.nn.relu(x)

        x = tf.layers.conv2d(
            inputs=x,
            filters=2*C,
            kernel_size=kernel,
            strides = (1,1),
            use_bias = False,
            padding="same")
        b = tf.Variable(tf.random_normal([2*C]))
        x = tf.nn.bias_add(x, b)
        gate = tf.nn.sigmoid(x)

        fused_feats = feats_con*gate
        fused_feats = tf.layers.conv2d(
            inputs=fused_feats,
            filters=C,
            kernel_size=kernel,
            strides = (1,1),
            use_bias = False,
            padding="same")
        b = tf.Variable(tf.random_normal([C]))
        fused_feats = tf.nn.bias_add(fused_feats, b)
        fused_feats = tf.layers.batch_normalization(fused_feats, training=True)
        fused_feats = tf.nn.relu(fused_feats)

    return fused_feats

# similar to SSMA but without the upper branch
def fuse_SSMA_like_concat(scope, feats_mod_1, feats_mod_2, training, C=512):
    kernel = [3, 3]
    eta = 16
    C_over_eta = 32

    with tf.variable_scope(scope):
        #feats_shape = tf.shape(feats_mod_1) # assuming both modality feats has same number of channels
        #C = feats_shape[3]
        fused_feats = tf.concat([feats_mod_1, feats_mod_2], 3)
        #fused_feats = feats_con*gate
        fused_feats = tf.layers.conv2d(
            inputs=fused_feats,
            filters=C,
            kernel_size=kernel,
            strides = (1,1),
            use_bias = False,
            padding="same")
        b = tf.Variable(tf.random_normal([C]))
        fused_feats = tf.nn.bias_add(fused_feats, b)
        fused_feats = tf.layers.batch_normalization(fused_feats, training=True)
        fused_feats = tf.nn.relu(fused_feats)

    return fused_feats

# similar to SSMA but without the upper branch and with summation not concat
def fuse_SSMA_like_sum(scope, feats_mod_1, feats_mod_2, training, C=512):
    kernel = [3, 3]
    eta = 16
    C_over_eta = 32

    with tf.variable_scope(scope):
        #feats_shape = tf.shape(feats_mod_1) # assuming both modality feats has same number of channels
        #C = feats_shape[3]
        fused_feats = feats_mod_1 + feats_mod_2

        fused_feats = tf.layers.conv2d(
            inputs=fused_feats,
            filters=C,
            kernel_size=kernel,
            strides = (1,1),
            use_bias = False,
            padding="same")
        b = tf.Variable(tf.random_normal([C]))
        fused_feats = tf.nn.bias_add(fused_feats, b)
        fused_feats = tf.layers.batch_normalization(fused_feats, training=True)
        fused_feats = tf.nn.relu(fused_feats)

    return fused_feats



def log_summaries(in_imgs, num_classes, logits, labels,loss):
    cityscapes_label_colormap = get_dataset_colormap.create_cityscapes_label_colormap()
    cmp = tf.convert_to_tensor(cityscapes_label_colormap, tf.int32)  # (256, 3)
    #pixel_scaling = max(1, 255 // num_classes)
    predictions = tf.expand_dims(tf.argmax(logits, 3), -1)
    summary_predictions = tf.gather(params=cmp, indices=predictions[:,:, :,0])
    summary_label = tf.gather(params=cmp, indices=labels[:,:, :,0])
    #summary_predictions = tf.cast(predictions * pixel_scaling, tf.uint8)
    #summary_label       = tf.cast(labels * pixel_scaling, tf.uint8)
    #colored_label = get_dataset_colormap.label_to_color_image(prediction, 'cityscapes')
    image = tf.cast(summary_predictions, tf.uint8)
    image_label = tf.cast(summary_label, tf.uint8)

    tf.summary.image('input_image', in_imgs)
    tf.summary.image("semantic_map", image)
    tf.summary.image("label", image_label)
    tf.summary.scalar('loss', loss)


def train():
    # From build_cityscapes_data.py: example = image_data, filename, height, width, seg_data
    tf.logging.set_verbosity(tf.logging.INFO)
    clone_batch_size = FLAGS.train_batch_size

    dataset = data_generator.Dataset(
        dataset_name=FLAGS.dataset,
        split_name=FLAGS.train_split,
        dataset_dir=FLAGS.dataset_dir,
        batch_size=clone_batch_size,
        crop_size=[int(sz) for sz in FLAGS.train_crop_size],
        min_resize_value=FLAGS.min_resize_value,
        max_resize_value=FLAGS.max_resize_value,
        resize_factor=FLAGS.resize_factor,
        min_scale_factor=FLAGS.min_scale_factor,
        max_scale_factor=FLAGS.max_scale_factor,
        scale_factor_step_size=FLAGS.scale_factor_step_size,
        model_variant=None,
        num_readers=10,
        is_training=True,
        should_shuffle=True,
        should_repeat=True)

    # reading batch: keys of samples ['height', 'width', 'image_name', 'label', 'image']
    num_classes = dataset.num_of_classes
    samples = dataset.get_one_shot_iterator().get_next()
    in_imgs = samples['image'] / 255
    labels  = samples['label']  #channel=1

    latents, skip = seg_encoder('Encoder', in_imgs, training= True)
    if FLAGS.use_skip_1by1:
        skip = tf.layers.conv2d(inputs=skip,filters=32,kernel_size=[1,1],strides = (1,1),use_bias = False,padding="same")
    if not FLAGS.use_skip:
        skip = None

    logits  = seg_decoder('Decoder', latents, training= True, num_classes=num_classes, skip=skip)

    #train_loss, _, _ = normal_loss(logits, labels, num_classes, dataset.ignore_label)
    train_loss = softmax_cross_entropy_loss_mining(logits,
                                      labels,
                                      num_classes,
                                      dataset.ignore_label,
                                      loss_weight=1.0,
                                      upsample_logits=False,
                                      hard_example_mining_step=FLAGS.hard_example_mining_step,
                                      top_k_percent_pixels=FLAGS.top_k_percent_pixels,
                                      scope='CI_Loss')

    log_summaries(in_imgs, num_classes, logits, labels, train_loss)
    step = tf.train.get_or_create_global_step()
    main_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4) #1e-4/100ksteps/bs 2  ==> same but lr = 1e-5
    main_step = main_optimizer.minimize(train_loss, global_step=step)
    train_op = tf.group(main_step)

    hooks = [
      tf.train.StopAtStepHook(last_step=FLAGS.last_step),
      tf.train.NanTensorHook(train_loss),]

    step_c = 0
    with tf.train.MonitoredTrainingSession(hooks=hooks, checkpoint_dir=FLAGS.checkpoint_dir,
      save_checkpoint_secs=300, save_summaries_secs=60) as sess:
        while not sess.should_stop():
            sess.run(train_op)



def main(unused_argv):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    train()


if __name__ == '__main__':
    tf.app.run()
