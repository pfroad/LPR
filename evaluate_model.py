# coding=utf-8
import tensorflow as tf
import numpy as np
import os

import data.input_data as ind

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import model
from google.protobuf import text_format as pbtf

index = {"京": 0, "沪": 1, "津": 2, "渝": 3, "冀": 4, "晋": 5, "蒙": 6, "辽": 7, "吉": 8, "黑": 9, "苏": 10, "浙": 11, "皖": 12,
         "闽": 13, "赣": 14, "鲁": 15, "豫": 16, "鄂": 17, "湘": 18, "粤": 19, "桂": 20, "琼": 21, "川": 22, "贵": 23, "云": 24,
         "藏": 25, "陕": 26, "甘": 27, "青": 28, "宁": 29, "新": 30, "0": 31, "1": 32, "2": 33, "3": 34, "4": 35, "5": 36,
         "6": 37, "7": 38, "8": 39, "9": 40, "A": 41, "B": 42, "C": 43, "D": 44, "E": 45, "F": 46, "G": 47, "H": 48,
         "J": 49, "K": 50, "L": 51, "M": 52, "N": 53, "P": 54, "Q": 55, "R": 56, "S": 57, "T": 58, "U": 59, "V": 60,
         "W": 61, "X": 62, "Y": 63, "Z": 64}

chars = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
         "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A",
         "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
         "Y", "Z"
         ]

batch_size = 10
img_w = 272
img_h = 72
channel = 3


def get_one_img(test):
    n = len(test)
    ind = np.random.randint(0, n)
    img_dir = test[ind]

    image_show = Image.open(img_dir)
    plt.imshow(image_show)
    # plt.show()
    image = cv2.imread(img_dir)
    img = np.multiply(image, 1 / 255)
    return np.array([img])


def get_batch():
    data_batch = ind.TrainData(batch_size, img_h, img_w, "./env_images", "./images")
    plates, labels = data_batch.data()
    return np.array(plates), np.array(labels)

pls, lbs = get_batch()
print(lbs)

x = tf.placeholder(tf.float32, [batch_size, img_h, img_w, channel])
keep_prob = tf.placeholder(tf.float32)

test_dir = './plates'
test_imgs = []

for file in os.listdir(test_dir):
    test_imgs.append(test_dir + "/" + file)
test_imgs = list(test_imgs)


image_array = get_one_img(test_imgs)


def load_graph(model_pb, input_map):
    f = tf.gfile.FastGFile(model_pb, 'rb')
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    persisted_graph = tf.import_graph_def(graph_def, input_map, name='')
    return persisted_graph


def load_txt_graph(model_pb):
    graph_def = tf.GraphDef()
    with open(dir, 'r') as fh:
        graph_str = fh.read()
    pbtf.Parse(graph_str, graph_def)
    persisted_graph = tf.import_graph_def(graph_def, name='')
    return persisted_graph


# logits1, logits2, logits3, logits4, logits5, logits6, logits7 = model.inference(x, keep_prob)

with tf.Session(graph=load_graph("./train_logs_50000/pr-model.pb", {"Placeholder:0": x})) as sess:
    # sess.run(tf.global_variables_initializer())
    logits2 = sess.graph.get_tensor_by_name("fc22/fc22:0")
    logits1 = sess.graph.get_tensor_by_name("fc21/fc21:0")
    logits3 = sess.graph.get_tensor_by_name("fc23/fc23:0")
    logits4 = sess.graph.get_tensor_by_name("fc24/fc24:0")
    logits5 = sess.graph.get_tensor_by_name("fc25/fc25:0")
    logits6 = sess.graph.get_tensor_by_name("fc26/fc26:0")
    logits7 = sess.graph.get_tensor_by_name("fc27/fc27:0")


    print(sess.graph.get_operation_by_name("fc21/fc21"))

    pre1, pre2, pre3, pre4, pre5, pre6, pre7 = sess.run([logits1, logits2, logits3, logits4, logits5, logits6, logits7],
                                                        feed_dict={x: np.reshape(image_array, [-1, 72, 272, 3]),
                                                                   keep_prob: 1.0})

    print(pre1)

    prediction = np.reshape(np.array([pre1[0], pre2[0], pre3[0], pre4[0], pre5[0], pre6[0], pre7[0]]), [-1, 65])
    # print(prediction)

    max_index = np.argmax(prediction, axis=1)
    print(max_index)
    # print(np.argmax(prediction, axis=0))

    line = ''
    for i in range(prediction.shape[0]):
        if i == 0:
            result = np.argmax(prediction[i][0:31])
        if i == 1:
            result = np.argmax(prediction[i][41:65]) + 41
        if i > 1:
            result = np.argmax(prediction[i][31:65]) + 31

        line += chars[result] + " "

    print('predicted: ' + line)
