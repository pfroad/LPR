# coding:utf-8

import numpy as np
import cv2
import data.genplate as gp


class TrainData:
    def __init__(self, batch_size, height, width, env_image, gen_images):
        self.genplate = gp.GenPlate("platech.ttf", "platechar.ttf", env_image, gen_images)
        self.batch_size = batch_size
        self.height = height
        self.width = width

    def data(self):
        data = []
        res = []
        for i in range(self.batch_size):
            num, img = self.gen_sample()
            data.append(img)
            data.append(num)
            res.append(data)

        return res

    def rand_range(self, lo, hi):
        return lo + gp.random(hi - lo)

    def gen_num(self):
        label = []
        label.append(self.rand_range(0, 31))
        label.append(self.rand_range(41, 65))
        for i in range(5):
            label.append(self.rand_range(31, 65))

        return "".join(gp.chars[s] for s in label), label

    def gen_sample(self):
        num, label = self.gen_num()
        img = cv2.resize(self.genplate.generate(num), (self.width, self.height))
        return label, np.multiply(img, 1 / 255.0)

# img_w = 272
# img_h = 72
# data_batch = TrainData(2, img_h, img_w)
# imgs, labels = data_batch.data()
#
# print(np.array(imgs).shape)