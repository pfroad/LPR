import os
import numpy as np
import tensorflow as tf

import data.input_data as ind
import model

import time
import datetime

img_w = 272
img_h = 72
num_label = 7
batch_size = 8
count = 30000
learning_rate = 0.0001
channel = 3

image_holder = tf.placeholder(tf.float32, [batch_size, img_h, img_w, channel])
label_holder = tf.placeholder(tf.int32, [batch_size, num_label])
keep_prob_val = 0.5
keep_prob =tf.placeholder(tf.float32)

logs_train_dir = './train_logs_50000/'


def get_batch():
    data_batch = ind.TrainData(batch_size, img_h, img_w, "./env_images", "./images")
    images, labels = data_batch.data()

    return np.array(images), np.array(labels)


logits1, logits2, logits3, logits4, logits5, logits6, logits7 = model.inference(image_holder, keep_prob_val)
loss1, loss2, loss3, loss4, loss5, loss6, loss7 = model.loss(logits1, logits2, logits3, logits4, logits5, logits6, logits7, label_holder)
op1, op2, op3, op4, op5, op6, op7 = model.training(loss1, loss2, loss3, loss4, loss5, loss6, loss7, learning_rate)

train_acc = model.evaluation(logits1, logits2, logits3, logits4, logits5, logits6, logits7, label_holder)

input_image = tf.summary.image('input', image_holder)
sum_op = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))

sess = tf.Session()
train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

start_time = time.time()

for step in range(count):
    imgs, labels = get_batch()
    # print(imgs)

    s2 = time.time()

    feed_dict = {image_holder: imgs, label_holder: labels, keep_prob: keep_prob_val}
    _, _, _, _, _, _, _, tra_loss1, tra_loss2, tra_loss3, tra_loss4, tra_loss5, tra_loss6, tra_loss7, acc, summary_str \
        = sess.run([op1, op2, op3, op4, op5, op6, op7, loss1, loss2, loss3, loss4, loss5, loss6, loss7, train_acc, sum_op], feed_dict)

    train_writer.add_summary(summary_str, step)
    losses = tra_loss1 + tra_loss2 + tra_loss3 + tra_loss4 + tra_loss5 + tra_loss6 + tra_loss7

    if step % 10 == 0:
        print('%s : Step %d,train_loss = %.2f,acc= %.2f,sec/batch=%.3f' %
              (datetime.datetime.now().isoformat(), step, losses, acc, float(time.time() - time.time())))

    if step % 10000 == 0 or step + 1 == count:
        checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
        saver = tf.train.Saver()
        saver.save(sess, checkpoint_path, global_step=step)

sess.close()

print('Training takes sec/batch=%.3f' % float(time.time() - start_time))
