import tensorflow as tf
import numpy as np


# from keras.models import Model
# from keras.callbacks import ModelCheckpoint
#
#
# # %matplotlib as inline
# np.random.seed(5)
# config = tf.ConfigProto()
#
# set_s
def inference(images, keep_prob):
    with tf.variable_scope("conv1") as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 3, 32],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable("bias",
                                 shape=[32],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv1 = tf.nn.relu(
            tf.nn.bias_add(
                tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='VALID'),
                biases),
            name='conv1')

    with tf.variable_scope("conv2") as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 32, 32],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable("bias",
                                 shape=[32],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv2 = tf.nn.relu(
            tf.nn.bias_add(
                tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='VALID'),
                biases),
            name='conv2')

    with tf.variable_scope("max_pooling1") as scope:
        pool1 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pooling1')

    with tf.variable_scope("conv3") as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 32, 64],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable("bias",
                                 shape=[64],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv3 = tf.nn.relu(
            tf.nn.bias_add(
                tf.nn.conv2d(pool1, weights, strides=[1, 1, 1, 1], padding='VALID'),
                biases),
            name='conv3')

    with tf.variable_scope("conv4") as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 64, 64],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable("bias",
                                 shape=[64],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv4 = tf.nn.relu(
            tf.nn.bias_add(
                tf.nn.conv2d(conv3, weights, strides=[1, 1, 1, 1], padding='VALID'),
                biases),
            name='conv4')

    with tf.variable_scope("max_pooling2") as scope:
        pool2 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pooling2')

    with tf.variable_scope("conv5") as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 64, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable("bias",
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv5 = tf.nn.relu(
            tf.nn.bias_add(
                tf.nn.conv2d(pool2, weights, strides=[1, 1, 1, 1], padding='VALID'),
                biases),
            name='conv5')

    with tf.variable_scope("conv6") as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 128, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable("bias",
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv6 = tf.nn.relu(
            tf.nn.bias_add(
                tf.nn.conv2d(conv5, weights, strides=[1, 1, 1, 1], padding='VALID'),
                biases),
            name='conv6')

    with tf.variable_scope("max_pooling3") as scope:
        pool3 = tf.nn.max_pool(conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pooling3')

    with tf.variable_scope("fc1") as scope:
        shape = pool3.get_shape()
        flattened_shape = shape[1].value * shape[2].value * shape[3].value
        reshape = tf.reshape(pool3, [-1, flattened_shape])

        fc1 = tf.nn.dropout(reshape, keep_prob, name="fc1_dropdot")

    with tf.variable_scope("fc21") as scope:
        weights = tf.get_variable("weights",
                                  shape=[flattened_shape, 65],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[65],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(0.1))
        fc21 = tf.add(tf.matmul(fc1, weights), biases, name="fc21")

    with tf.variable_scope("fc22") as scope:
        weights = tf.get_variable("weights",
                                  shape=[flattened_shape, 65],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[65],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(0.1))
        fc22 = tf.add(tf.matmul(fc1, weights), biases, name="fc22")

    with tf.variable_scope("fc23") as scope:
        weights = tf.get_variable("weights",
                                  shape=[flattened_shape, 65],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[65],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(0.1))
        fc23 = tf.add(tf.matmul(fc1, weights), biases, name="fc23")

    with tf.variable_scope("fc24") as scope:
        weights = tf.get_variable("weights",
                                  shape=[flattened_shape, 65],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[65],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(0.1))
        fc24 = tf.add(tf.matmul(fc1, weights), biases, name="fc24")

    with tf.variable_scope("fc25") as scope:
        weights = tf.get_variable("weights",
                                  shape=[flattened_shape, 65],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[65],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(0.1))
        fc25 = tf.add(tf.matmul(fc1, weights), biases, name="fc25")

    with tf.variable_scope("fc26") as scope:
        weights = tf.get_variable("weights",
                                  shape=[flattened_shape, 65],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[65],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(0.1))
        fc26 = tf.add(tf.matmul(fc1, weights), biases, name="fc26")

    with tf.variable_scope("fc27") as scope:
        weights = tf.get_variable("weights",
                                  shape=[flattened_shape, 65],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[65],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(0.1))
        fc27 = tf.add(tf.matmul(fc1, weights), biases, name="fc27")

    return fc21, fc22, fc23, fc24, fc25, fc26, fc27


def loss(logits1, logits2, logits3, logits4, logits5, logits6, logits7, labels):
    labels = tf.convert_to_tensor(labels, tf.int32)
    with tf.variable_scope('loss1') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits1, labels=labels[:, 0],
                                                                       name='xentropy_per_example')
        # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels,name='xentropy_per_example')
        loss1 = tf.reduce_mean(cross_entropy, name='loss1')
        tf.summary.scalar(scope.name + '/loss1', loss1)

    with tf.variable_scope('loss2') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits2, labels=labels[:, 1],
                                                                       name='xentropy_per_example')
        # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels,name='xentropy_per_example')
        loss2 = tf.reduce_mean(cross_entropy, name='loss2')
        tf.summary.scalar(scope.name + '/loss2', loss2)

    with tf.variable_scope('loss3') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits3, labels=labels[:, 2],
                                                                       name='xentropy_per_example')
        # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels,name='xentropy_per_example')
        loss3 = tf.reduce_mean(cross_entropy, name='loss3')
        tf.summary.scalar(scope.name + '/loss3', loss3)

    with tf.variable_scope('loss4') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits4, labels=labels[:, 3],
                                                                       name='xentropy_per_example')
        # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels,name='xentropy_per_example')
        loss4 = tf.reduce_mean(cross_entropy, name='loss4')
        tf.summary.scalar(scope.name + '/loss4', loss4)

    with tf.variable_scope('loss5') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits5, labels=labels[:, 4],
                                                                       name='xentropy_per_example')
        # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels,name='xentropy_per_example')
        loss5 = tf.reduce_mean(cross_entropy, name='loss5')
        tf.summary.scalar(scope.name + '/loss5', loss5)

    with tf.variable_scope('loss6') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits6, labels=labels[:, 5],
                                                                       name='xentropy_per_example')
        # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels,name='xentropy_per_example')
        loss6 = tf.reduce_mean(cross_entropy, name='loss6')
        tf.summary.scalar(scope.name + '/loss6', loss6)

    with tf.variable_scope('loss7') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits7, labels=labels[:, 6],
                                                                       name='xentropy_per_example')
        # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels,name='xentropy_per_example')
        loss7 = tf.reduce_mean(cross_entropy, name='loss7')
        tf.summary.scalar(scope.name + '/loss7', loss7)

    return loss1, loss2, loss3, loss4, loss5, loss6, loss7


def training(loss1, loss2, loss3, loss4, loss5, loss6, loss7, learning_rate):
    '''Training ops, the Op returned by this function is what must be passed to
        'sess.run()' call to cause the model to train.

    Args:
        loss: loss tensor, from losses()

    Returns:
        train_op: The op for trainning
    '''
    with tf.name_scope('optimizer1'):
        optimizer1 = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op1 = optimizer1.minimize(loss1, global_step=global_step)
    with tf.name_scope('optimizer2'):
        optimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op2 = optimizer2.minimize(loss2, global_step=global_step)
    with tf.name_scope('optimizer3'):
        optimizer3 = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op3 = optimizer3.minimize(loss3, global_step=global_step)
    with tf.name_scope('optimizer4'):
        optimizer4 = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op4 = optimizer4.minimize(loss4, global_step=global_step)
    with tf.name_scope('optimizer5'):
        optimizer5 = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op5 = optimizer5.minimize(loss5, global_step=global_step)
    with tf.name_scope('optimizer6'):
        optimizer6 = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op6 = optimizer6.minimize(loss6, global_step=global_step)
    with tf.name_scope('optimizer7'):
        optimizer7 = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op7 = optimizer7.minimize(loss7, global_step=global_step)

    return train_op1, train_op2, train_op3, train_op4, train_op5, train_op6, train_op7


def evaluation(logits1, logits2, logits3, logits4, logits5, logits6, logits7, labels):
    """Evaluate the quality of the logits at predicting the label.
      Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size], with values in the
          range [0, NUM_CLASSES).
      Returns:
        A scalar int32 tensor with the number of examples (out of batch_size)
        that were predicted correctly.
      """
    logits_all = tf.concat([logits1, logits2, logits3, logits4, logits5, logits6, logits7], 0)
    labels = tf.convert_to_tensor(labels, tf.int32)
    labels_all = tf.reshape(tf.transpose(labels), [-1])
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits_all, labels_all, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy
