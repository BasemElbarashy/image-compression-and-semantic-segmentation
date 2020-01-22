# Arch
"""
comp_numfilters = 128
y = analysis_transform(x_rgb, comp_numfilters)
entropy_bottleneck = tfc.EntropyBottleneck()
y_tilde_hat, likelihoods = entropy_bottleneck(y, training=istraining)
x_tilde_hat = synthesis_transform(y_tilde_hat, comp_numfilters)
if mode == 'testing':
    string = entropy_bottleneck.compress(y)
    string = tf.squeeze(string, axis=0)
"""

# seg_comp_rgb
def build_model(x_rgb, x_depth, lmbda, num_classes, mode = 'training', seg_labels = None,
                ignore_label = None):
    # Args
    train_loss, bpp, mse, y_tilde_hat, x_tilde_hat, y, string, entropy_bottleneck, seg_logits = [None]*9
    seg_loss = None
    istraining = (mode == 'training')
    num_pixels = tf.to_float(tf.reduce_prod(tf.shape(x_rgb)[:-1]))

    y, skip = seg_encoder('Encoder', x_rgb, training=istraining)
    entropy_bottleneck = tfc.EntropyBottleneck()
    y_tilde_hat, likelihoods = entropy_bottleneck(y, training=istraining)
    skip = None
    seg_logits = seg_decoder('Decoder', y_tilde_hat, training=istraining, num_classes=num_classes, skip=skip)
    if mode == 'testing':
        string = entropy_bottleneck.compress(y)
        string = tf.squeeze(string, axis=0)


    # Loss

    bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)
    #mse = tf.reduce_mean(tf.squared_difference(x_rgb, x_tilde_hat))

    if istraining:
        seg_loss = softmax_cross_entropy_loss_mining(seg_logits,
                                                     seg_labels,
                                                     num_classes,
                                                     ignore_label,
                                                     loss_weight=1.0,
                                                     upsample_logits=False,
                                                     hard_example_mining_step=100000,
                                                     top_k_percent_pixels=0.2,
                                                     scope='CI_Loss')
        train_loss = lmbda * seg_loss + bpp
    return train_loss, bpp, mse, x_tilde_hat, y_tilde_hat, y, string, entropy_bottleneck, seg_logits, seg_loss

# seg_comp_depth
def build_model(x_rgb, x_depth, lmbda, num_classes, mode = 'training', seg_labels = None,
                ignore_label = None):
    # Args
    train_loss, bpp, mse, y_tilde_hat, x_tilde_hat, y, string, entropy_bottleneck, seg_logits = [None]*9
    seg_loss = None
    istraining = (mode == 'training')
    num_pixels = tf.to_float(tf.reduce_prod(tf.shape(x_rgb)[:-1]))

    y, skip = seg_encoder('Encoder', x_depth, training=istraining)
    entropy_bottleneck = tfc.EntropyBottleneck()
    y_tilde_hat, likelihoods = entropy_bottleneck(y, training=istraining)
    skip = None
    seg_logits = seg_decoder('Decoder', y_tilde_hat, training=istraining, num_classes=num_classes, skip=skip)
    if mode == 'testing':
        string = entropy_bottleneck.compress(y)
        string = tf.squeeze(string, axis=0)


    # Loss

    bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)
    #mse = tf.reduce_mean(tf.squared_difference(x_rgb, x_tilde_hat))

    if istraining:
        seg_loss = softmax_cross_entropy_loss_mining(seg_logits,
                                                     seg_labels,
                                                     num_classes,
                                                     ignore_label,
                                                     loss_weight=1.0,
                                                     upsample_logits=False,
                                                     hard_example_mining_step=100000,
                                                     top_k_percent_pixels=0.2,
                                                     scope='CI_Loss')
        train_loss = lmbda * seg_loss + bpp
    return train_loss, bpp, mse, x_tilde_hat, y_tilde_hat, y, string, entropy_bottleneck, seg_logits, seg_loss


# seg_comp_rgb_d
def build_model(x_rgb, x_depth, lmbda, num_classes, mode = 'training', seg_labels = None,
                ignore_label = None):
    # Args
    train_loss, bpp, mse, y_tilde_hat, x_tilde_hat, y, string, entropy_bottleneck, seg_logits = [None]*9
    seg_loss = None
    istraining = (mode == 'training')
    num_pixels = tf.to_float(tf.reduce_prod(tf.shape(x_rgb)[:-1]))

    # arch
    rgb_feats, _ = seg_encoder('Encoder_rgb', x_rgb, training=istraining)
    depth_feats, _ = seg_encoder('Encoder_depth', x_depth, training=istraining)
    y = fuse_SSMA('latent_fuse', rgb_feats, depth_feats, training=istraining, C=512)

    entropy_bottleneck = tfc.EntropyBottleneck()
    y_tilde_hat, likelihoods = entropy_bottleneck(y, training=istraining)
    skip = None
    seg_logits = seg_decoder('Decoder', y_tilde_hat, training=istraining, num_classes=num_classes, skip=skip)
    if mode == 'testing':
        string = entropy_bottleneck.compress(y)
        string = tf.squeeze(string, axis=0)


    # Loss

    bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)
    #mse = tf.reduce_mean(tf.squared_difference(x_rgb, x_tilde_hat))

    if istraining:
        seg_loss = softmax_cross_entropy_loss_mining(seg_logits,
                                                     seg_labels,
                                                     num_classes,
                                                     ignore_label,
                                                     loss_weight=1.0,
                                                     upsample_logits=False,
                                                     hard_example_mining_step=100000,
                                                     top_k_percent_pixels=0.2,
                                                     scope='CI_Loss')
        train_loss = lmbda * seg_loss + bpp
    return train_loss, bpp, mse, x_tilde_hat, y_tilde_hat, y, string, entropy_bottleneck, seg_logits, seg_loss



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


#seg_wocomp_depth
def build_model(x_rgb, x_depth, lmbda, num_classes, mode = 'training', seg_labels = None, ignore_label = None):
    # Args
    train_loss, bpp, mse, y_tilde_hat, x_tilde_hat, y, string, entropy_bottleneck, seg_logits = [None]*9
    seg_loss = None
    istraining = (mode == 'training')
    num_pixels = tf.to_float(tf.reduce_prod(tf.shape(x_rgb)[:-1]))


    y, _ = seg_encoder('Encoder_depth', x_depth, training=istraining)

    skip = None
    seg_logits     = seg_decoder('Decoder', y, training=istraining, num_classes=num_classes, skip=skip)

    # Loss

    if istraining:
        seg_loss = softmax_cross_entropy_loss_mining(seg_logits,
                                                     seg_labels,
                                                     num_classes,
                                                     ignore_label,
                                                     loss_weight=1.0,
                                                     upsample_logits=False,
                                                     hard_example_mining_step=100000,
                                                     top_k_percent_pixels=0.2,
                                                     scope='CI_Loss')
        train_loss =  seg_loss

    return train_loss, bpp, mse, x_tilde_hat, y_tilde_hat, y, string, entropy_bottleneck, seg_logits, seg_loss

# seg_wocomp_rgbd
def build_model(x_rgb, x_depth, lmbda, num_classes, mode = 'training', seg_labels = None, ignore_label = None):
    # Args
    train_loss, bpp, mse, y_tilde_hat, x_tilde_hat, y, string, entropy_bottleneck, seg_logits = [None]*9
    seg_loss = None
    istraining = (mode == 'training')
    num_pixels = tf.to_float(tf.reduce_prod(tf.shape(x_rgb)[:-1]))

    rgb_feats, _   = seg_encoder('Encoder_rgb', x_rgb, training=istraining)
    depth_feats, _ = seg_encoder('Encoder_depth', x_depth, training=istraining)
    y              = fuse_SSMA('latent_fuse', rgb_feats, depth_feats, training=istraining, C=512)
    skip = None
    seg_logits     = seg_decoder('Decoder', y, training=istraining, num_classes=num_classes, skip=skip)

    # Loss

    if istraining:
        seg_loss = softmax_cross_entropy_loss_mining(seg_logits,
                                                     seg_labels,
                                                     num_classes,
                                                     ignore_label,
                                                     loss_weight=1.0,
                                                     upsample_logits=False,
                                                     hard_example_mining_step=100000,
                                                     top_k_percent_pixels=0.2,
                                                     scope='CI_Loss')
        train_loss =  seg_loss

    return train_loss, bpp, mse, x_tilde_hat, y_tilde_hat, y, string, entropy_bottleneck, seg_logits, seg_loss

# SSMA like concatenation
# seg_wocomp_rgbd_concat
def build_model(x_rgb, x_depth, lmbda, num_classes, mode = 'training', seg_labels = None, ignore_label = None):
    # Args
    train_loss, bpp, mse, y_tilde_hat, x_tilde_hat, y, string, entropy_bottleneck, seg_logits = [None]*9
    seg_loss = None
    istraining = (mode == 'training')
    num_pixels = tf.to_float(tf.reduce_prod(tf.shape(x_rgb)[:-1]))

    rgb_feats, _   = seg_encoder('Encoder_rgb', x_rgb, training=istraining)
    depth_feats, _ = seg_encoder('Encoder_depth', x_depth, training=istraining)
    y              = fuse_SSMA_like_concat('latent_fuse', rgb_feats, depth_feats, training=istraining, C=512)
    skip = None
    seg_logits     = seg_decoder('Decoder', y, training=istraining, num_classes=num_classes, skip=skip)

    # Loss

    if istraining:
        seg_loss = softmax_cross_entropy_loss_mining(seg_logits,
                                                     seg_labels,
                                                     num_classes,
                                                     ignore_label,
                                                     loss_weight=1.0,
                                                     upsample_logits=False,
                                                     hard_example_mining_step=100000,
                                                     top_k_percent_pixels=0.2,
                                                     scope='CI_Loss')
        train_loss =  seg_loss

    return train_loss, bpp, mse, x_tilde_hat, y_tilde_hat, y, string, entropy_bottleneck, seg_logits, seg_loss


# SSMA like summation
# seg_wocomp_rgbd_sum
def build_model(x_rgb, x_depth, lmbda, num_classes, mode = 'training', seg_labels = None, ignore_label = None):
    # Args
    train_loss, bpp, mse, y_tilde_hat, x_tilde_hat, y, string, entropy_bottleneck, seg_logits = [None]*9
    seg_loss = None
    istraining = (mode == 'training')
    num_pixels = tf.to_float(tf.reduce_prod(tf.shape(x_rgb)[:-1]))

    rgb_feats, _   = seg_encoder('Encoder_rgb', x_rgb, training=istraining)
    depth_feats, _ = seg_encoder('Encoder_depth', x_depth, training=istraining)
    y              = fuse_SSMA_like_sum('latent_fuse', rgb_feats, depth_feats, training=istraining, C=512)
    skip = None
    seg_logits     = seg_decoder('Decoder', y, training=istraining, num_classes=num_classes, skip=skip)

    # Loss

    if istraining:
        seg_loss = softmax_cross_entropy_loss_mining(seg_logits,
                                                     seg_labels,
                                                     num_classes,
                                                     ignore_label,
                                                     loss_weight=1.0,
                                                     upsample_logits=False,
                                                     hard_example_mining_step=100000,
                                                     top_k_percent_pixels=0.2,
                                                     scope='CI_Loss')
        train_loss =  seg_loss

    return train_loss, bpp, mse, x_tilde_hat, y_tilde_hat, y, string, entropy_bottleneck, seg_logits, seg_loss


# seg_com_rgb_gdn
def build_model(x_rgb, x_depth, lmbda, num_classes, mode = 'training', seg_labels = None,
                ignore_label = None):
    # Args
    train_loss, bpp, mse, y_tilde_hat, x_tilde_hat, y, string, entropy_bottleneck, seg_logits = [None]*9
    seg_loss = None
    istraining = (mode == 'training')
    num_pixels = tf.to_float(tf.reduce_prod(tf.shape(x_rgb)[:-1]))

    y, skip = seg_encoder_gdn('Encoder', x_rgb, training=istraining)
    entropy_bottleneck = tfc.EntropyBottleneck()
    y_tilde_hat, likelihoods = entropy_bottleneck(y, training=istraining)
    skip = None
    seg_logits = seg_decoder_gdn('Decoder', y_tilde_hat, training=istraining, num_classes=num_classes, skip=skip)
    if mode == 'testing':
        string = entropy_bottleneck.compress(y)
        string = tf.squeeze(string, axis=0)


    # Loss

    bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)
    #mse = tf.reduce_mean(tf.squared_difference(x_rgb, x_tilde_hat))

    if istraining:
        seg_loss = softmax_cross_entropy_loss_mining(seg_logits,
                                                     seg_labels,
                                                     num_classes,
                                                     ignore_label,
                                                     loss_weight=1.0,
                                                     upsample_logits=False,
                                                     hard_example_mining_step=100000,
                                                     top_k_percent_pixels=0.2,
                                                     scope='CI_Loss')
        train_loss = lmbda * seg_loss + bpp
    return train_loss, bpp, mse, x_tilde_hat, y_tilde_hat, y, string, entropy_bottleneck, seg_logits, seg_loss


#seg_com_reco_rgb_fixedlmbda2048
def build_model(x_rgb, x_depth, lmbda, num_classes, mode = 'training', seg_labels = None,
                ignore_label = None):
    # Args
    train_loss, bpp, mse, y_tilde_hat, x_tilde_hat, y, string, entropy_bottleneck, seg_logits = [None]*9
    seg_loss = None
    istraining = (mode == 'training')
    num_pixels = tf.to_float(tf.reduce_prod(tf.shape(x_rgb)[:-1]))

    y, skip = seg_encoder('Encoder', x_rgb, training=istraining)
    entropy_bottleneck = tfc.EntropyBottleneck()
    y_tilde_hat, likelihoods = entropy_bottleneck(y, training=istraining)
    skip = None

    seg_logits  = seg_decoder('Decoder_seg', y_tilde_hat, training=istraining, num_classes=num_classes, skip=skip)
    x_tilde_hat = seg_decoder('Decoder_rec', y_tilde_hat, training=istraining, num_classes=-1, skip=skip)
    if mode == 'testing':
        string = entropy_bottleneck.compress(y)
        string = tf.squeeze(string, axis=0)


    # Loss

    bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)
    mse = tf.reduce_mean(tf.squared_difference(x_rgb, x_tilde_hat))

    if istraining:
        seg_loss = softmax_cross_entropy_loss_mining(seg_logits,
                                                     seg_labels,
                                                     num_classes,
                                                     ignore_label,
                                                     loss_weight=1.0,
                                                     upsample_logits=False,
                                                     hard_example_mining_step=100000,
                                                     top_k_percent_pixels=0.2,
                                                     scope='CI_Loss')
        beta  = lmbda
        lmbda = 2048
        train_loss = lmbda *mse + beta* seg_loss + bpp
    return train_loss, bpp, mse, x_tilde_hat, y_tilde_hat, y, string, entropy_bottleneck, seg_logits, seg_loss



#seg_com_reco_rgb_fixedlmbda512
def build_model(x_rgb, x_depth, lmbda, num_classes, mode = 'training', seg_labels = None,
                ignore_label = None):
    # Args
    train_loss, bpp, mse, y_tilde_hat, x_tilde_hat, y, string, entropy_bottleneck, seg_logits = [None]*9
    seg_loss = None
    istraining = (mode == 'training')
    num_pixels = tf.to_float(tf.reduce_prod(tf.shape(x_rgb)[:-1]))

    y, skip = seg_encoder('Encoder', x_rgb, training=istraining)
    entropy_bottleneck = tfc.EntropyBottleneck()
    y_tilde_hat, likelihoods = entropy_bottleneck(y, training=istraining)
    skip = None

    seg_logits  = seg_decoder('Decoder_seg', y_tilde_hat, training=istraining, num_classes=num_classes, skip=skip)
    x_tilde_hat = seg_decoder('Decoder_rec', y_tilde_hat, training=istraining, num_classes=-1, skip=skip)
    if mode == 'testing':
        string = entropy_bottleneck.compress(y)
        string = tf.squeeze(string, axis=0)


    # Loss

    bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)
    mse = tf.reduce_mean(tf.squared_difference(x_rgb, x_tilde_hat))

    if istraining:
        seg_loss = softmax_cross_entropy_loss_mining(seg_logits,
                                                     seg_labels,
                                                     num_classes,
                                                     ignore_label,
                                                     loss_weight=1.0,
                                                     upsample_logits=False,
                                                     hard_example_mining_step=100000,
                                                     top_k_percent_pixels=0.2,
                                                     scope='CI_Loss')
        beta  = lmbda
        lmbda = 512
        train_loss = lmbda *mse + beta* seg_loss + bpp
    return train_loss, bpp, mse, x_tilde_hat, y_tilde_hat, y, string, entropy_bottleneck, seg_logits, seg_loss

#seg_com_reco_rgb_fixedlmbda8192
def build_model(x_rgb, x_depth, lmbda, num_classes, mode = 'training', seg_labels = None,
                ignore_label = None):
    # Args
    train_loss, bpp, mse, y_tilde_hat, x_tilde_hat, y, string, entropy_bottleneck, seg_logits = [None]*9
    seg_loss = None
    istraining = (mode == 'training')
    num_pixels = tf.to_float(tf.reduce_prod(tf.shape(x_rgb)[:-1]))

    y, skip = seg_encoder('Encoder', x_rgb, training=istraining)
    entropy_bottleneck = tfc.EntropyBottleneck()
    y_tilde_hat, likelihoods = entropy_bottleneck(y, training=istraining)
    skip = None

    seg_logits  = seg_decoder('Decoder_seg', y_tilde_hat, training=istraining, num_classes=num_classes, skip=skip)
    x_tilde_hat = seg_decoder('Decoder_rec', y_tilde_hat, training=istraining, num_classes=-1, skip=skip)
    if mode == 'testing':
        string = entropy_bottleneck.compress(y)
        string = tf.squeeze(string, axis=0)


    # Loss

    bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)
    mse = tf.reduce_mean(tf.squared_difference(x_rgb, x_tilde_hat))

    if istraining:
        seg_loss = softmax_cross_entropy_loss_mining(seg_logits,
                                                     seg_labels,
                                                     num_classes,
                                                     ignore_label,
                                                     loss_weight=1.0,
                                                     upsample_logits=False,
                                                     hard_example_mining_step=100000,
                                                     top_k_percent_pixels=0.2,
                                                     scope='CI_Loss')
        beta  = lmbda
        lmbda = 8192
        train_loss = lmbda *mse + beta* seg_loss + bpp
    return train_loss, bpp, mse, x_tilde_hat, y_tilde_hat, y, string, entropy_bottleneck, seg_logits, seg_loss

#seg_com_reco_rgb_fixedlmbda128
def build_model(x_rgb, x_depth, lmbda, num_classes, mode = 'training', seg_labels = None,
                ignore_label = None):
    # Args
    train_loss, bpp, mse, y_tilde_hat, x_tilde_hat, y, string, entropy_bottleneck, seg_logits = [None]*9
    seg_loss = None
    istraining = (mode == 'training')
    num_pixels = tf.to_float(tf.reduce_prod(tf.shape(x_rgb)[:-1]))

    y, skip = seg_encoder('Encoder', x_rgb, training=istraining)
    entropy_bottleneck = tfc.EntropyBottleneck()
    y_tilde_hat, likelihoods = entropy_bottleneck(y, training=istraining)
    skip = None

    seg_logits  = seg_decoder('Decoder_seg', y_tilde_hat, training=istraining, num_classes=num_classes, skip=skip)
    x_tilde_hat = seg_decoder('Decoder_rec', y_tilde_hat, training=istraining, num_classes=-1, skip=skip)
    if mode == 'testing':
        string = entropy_bottleneck.compress(y)
        string = tf.squeeze(string, axis=0)


    # Loss

    bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)
    mse = tf.reduce_mean(tf.squared_difference(x_rgb, x_tilde_hat))

    if istraining:
        seg_loss = softmax_cross_entropy_loss_mining(seg_logits,
                                                     seg_labels,
                                                     num_classes,
                                                     ignore_label,
                                                     loss_weight=1.0,
                                                     upsample_logits=False,
                                                     hard_example_mining_step=100000,
                                                     top_k_percent_pixels=0.2,
                                                     scope='CI_Loss')
        beta  = lmbda
        lmbda = 128
        train_loss = lmbda *mse + beta* seg_loss + bpp
    return train_loss, bpp, mse, x_tilde_hat, y_tilde_hat, y, string, entropy_bottleneck, seg_logits, seg_loss



#seg_com_reco_rgb_fixedlmbda16384
def build_model(x_rgb, x_depth, lmbda, num_classes, mode = 'training', seg_labels = None,
                ignore_label = None):
    # Args
    train_loss, bpp, mse, y_tilde_hat, x_tilde_hat, y, string, entropy_bottleneck, seg_logits = [None]*9
    seg_loss = None
    istraining = (mode == 'training')
    num_pixels = tf.to_float(tf.reduce_prod(tf.shape(x_rgb)[:-1]))

    y, skip = seg_encoder('Encoder', x_rgb, training=istraining)
    entropy_bottleneck = tfc.EntropyBottleneck()
    y_tilde_hat, likelihoods = entropy_bottleneck(y, training=istraining)
    skip = None

    seg_logits  = seg_decoder('Decoder_seg', y_tilde_hat, training=istraining, num_classes=num_classes, skip=skip)
    x_tilde_hat = seg_decoder('Decoder_rec', y_tilde_hat, training=istraining, num_classes=-1, skip=skip)
    if mode == 'testing':
        string = entropy_bottleneck.compress(y)
        string = tf.squeeze(string, axis=0)


    # Loss

    bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)
    mse = tf.reduce_mean(tf.squared_difference(x_rgb, x_tilde_hat))

    if istraining:
        seg_loss = softmax_cross_entropy_loss_mining(seg_logits,
                                                     seg_labels,
                                                     num_classes,
                                                     ignore_label,
                                                     loss_weight=1.0,
                                                     upsample_logits=False,
                                                     hard_example_mining_step=100000,
                                                     top_k_percent_pixels=0.2,
                                                     scope='CI_Loss')
        beta  = lmbda
        lmbda = 16384
        train_loss = lmbda *mse + beta* seg_loss + bpp
    return train_loss, bpp, mse, x_tilde_hat, y_tilde_hat, y, string, entropy_bottleneck, seg_logits, seg_loss



# seg_comp_rgb_d_downby2
def build_model(x_rgb, x_depth, lmbda, num_classes, mode = 'training', seg_labels = None,
                ignore_label = None):
    # Args
    train_loss, bpp, mse, y_tilde_hat, x_tilde_hat, y, string, entropy_bottleneck, seg_logits = [None]*9
    seg_loss = None
    istraining = (mode == 'training')
    num_pixels = tf.to_float(tf.reduce_prod(tf.shape(x_rgb)[:-1]))

    # arch
    rgb_feats, _ = seg_encoder('Encoder_rgb', x_rgb, training=istraining, add_sampling_layer=True)
    depth_feats, _ = seg_encoder('Encoder_depth', x_depth, training=istraining, add_sampling_layer=True)
    y = fuse_SSMA('latent_fuse', rgb_feats, depth_feats, training=istraining, C=512)

    entropy_bottleneck = tfc.EntropyBottleneck()
    y_tilde_hat, likelihoods = entropy_bottleneck(y, training=istraining)
    skip = None
    seg_logits = seg_decoder('Decoder', y_tilde_hat, training=istraining, num_classes=num_classes, skip=skip, add_sampling_layer=True)
    if mode == 'testing':
        string = entropy_bottleneck.compress(y)
        string = tf.squeeze(string, axis=0)


    # Loss

    bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)
    #mse = tf.reduce_mean(tf.squared_difference(x_rgb, x_tilde_hat))

    if istraining:
        seg_loss = softmax_cross_entropy_loss_mining(seg_logits,
                                                     seg_labels,
                                                     num_classes,
                                                     ignore_label,
                                                     loss_weight=1.0,
                                                     upsample_logits=False,
                                                     hard_example_mining_step=100000,
                                                     top_k_percent_pixels=0.2,
                                                     scope='CI_Loss')
        train_loss = lmbda * seg_loss + bpp
    return train_loss, bpp, mse, x_tilde_hat, y_tilde_hat, y, string, entropy_bottleneck, seg_logits, seg_loss
