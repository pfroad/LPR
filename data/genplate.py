# coding:utf-8

import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

import cv2
import numpy as np
import os
from math import *

# font = ImageFont.trueType("Arial-Bold.ttf", 14)
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


def rot(img, angel, shape, max_angel):
    size_o = [shape[1], shape[0]]
    size = (shape[1] + int(shape[0] * cos((float(max_angel) / 180) * 3.14)), shape[0])
    interval = abs(int(sin(float(angel / 180) * 3.14) * shape[0]))

    pts1 = np.float32([[0, 0], [0, size_o[1]], [size_o[0], 0], [size_o[0], size_o[1]]])

    if angel > 0:
        pts2 = np.float32([[interval, 0], [0, size[1]], [size[0], 0], [size[0] - interval, size_o[1]]])
    else:
        pts2 = np.float32([[0, 0], [interval, size[1]], [size[0] - interval, 0], [size[0], size_o[1]]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, size)

    return dst


def random(val):
    return int(np.random.random() * val)


def rotRandom(img, factor, size):
    # 使图像畸变
    # img 图像
    # factor 畸变参数
    # size图片的目标尺寸
    shape = size
    pts1 = np.float32([[0, 0], [0, shape[0]], [shape[1], 0], [shape[1], shape[0]]])
    pts2 = np.float32([[random(factor), random(factor)], [random(factor), shape[0] - random(factor)], [shape[1] - random(factor), random(factor)],
                       [shape[1] - random(factor), shape[0] - random(factor)]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, size)

    return dst


def tfactor(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hsv[:, :, 0] = hsv[:, :, 0] * (0.8 + np.random.random() * 0.2)
    hsv[:, :, 1] = hsv[:, :, 1] * (0.3 + np.random.random() * 0.7)
    hsv[:, :, 2] = hsv[:, :, 2] * (0.2 + np.random.random() * 0.8)

    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def random_environment(img, data_set):
    env = cv2.imread(data_set[random(len(data_set))])
    env = cv2.resize(env, (img.shape[1], img.shape[0]))

    bak = (img == 0).astype(np.uint8) * 255
    inv = cv2.bitwise_and(bak, env)
    img = cv2.bitwise_or(inv, img)
    return img


def gen_ch(font, val):
    img = Image.new("RGB", (45, 70), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 3), val, (0, 0, 0), font=font)
    img = img.resize((23, 70))
    return np.array(img)


def gen_eng(font, val):
    img = Image.new("RGB", (23, 70), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 2), val, (0, 0, 0), font=font)
    return np.array(img)


def addGauss(img, level):
    return cv2.blur(img, (level * 2 + 1, level * 2 + 1))


def addNoiseSingleChannel(ch):
    diff = 255 - ch.max()
    noise = np.random.normal(0, 1 + random(6), ch.shape)
    return ch + (diff * (noise - noise.min()) / (noise.max() - noise.min())).astype(np.uint8)


def addNoise(img, sdev=0.5, avg=10):
    img[:, :, 0] = addNoiseSingleChannel(img[:, :, 0])
    img[:, :, 0] = addNoiseSingleChannel(img[:, :, 0])
    img[:, :, 0] = addNoiseSingleChannel(img[:, :, 0])

    return img


class GenPlate:

    def __init__(self, font_ch, font_eng, plate_nos, images_path):
        self.font_eng = ImageFont.truetype(font_eng, 60, 0)
        self.font_ch = ImageFont.truetype(font_ch, 43, 0)
        self.plate_nos = plate_nos
        self.img = np.array(Image.new("RGB", (226, 70), (255, 255, 255)))
        self.bg = cv2.resize(cv2.imread(images_path + "/template.bmp"), (226, 70))
        self.smu = cv2.imread(images_path + "/images/smu2.jpg")
        self.plate_nos_path = []
        for parent, parent_folder, filenames in os.walk(plate_nos):
            for filename in filenames:
                path = parent + "/"+ filename
                self.plate_nos_path.append(path)

    def draw(self, val):
        offset = 2
        self.img[0: 70, offset + 8: offset + 8 + 23] = gen_ch(self.font_ch, val[0])
        self.img[0: 70, offset + 8 + 23 + 6: offset + 8 + 23 + 6 + 23] = gen_eng(self.font_eng, val[1])

        for i in range(5):
            base = offset + 8 + 23 + 6 + 23 + 17 + i * 23 + i * 6
            self.img[0: 70, base: base + 23] = gen_eng(self.font_eng, val[i + offset])
        return self.img

    def generate(self, text):
        if len(text) == 7:
            fg = cv2.bitwise_not(self.draw(text.encode('utf-8').decode(encoding="utf-8")))
            com = cv2.bitwise_or(fg, self.bg)
            com = rot(com, random(60) - 30, com.shape, 30)
            com = rotRandom(com, 10, (com.shape[1], com.shape[0]))

            com = tfactor(com)
            com = random_environment(com, self.plate_nos_path)
            return addNoise(addGauss(com, 1 + random(4)))

    def gen_plate_string(self, pos, val):
        plate_str = ""
        box = [0, 0, 0, 0, 0, 0, 0]
        if pos != -1:
            box[pos] = 1

        for unit, cpos in zip(box, range(len(box))):
            if unit == 1:
                plate_str += val
            else:
                if cpos == 0:
                    plate_str += chars[random(31)]
                elif cpos == 1:
                    plate_str += chars[41 + random(24)]
                else:
                    plate_str += chars[31 + random(34)]

        return plate_str

    def gen_batch(self, batch_size, pos, char_range, output_path, size):
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        for i in range(batch_size):
            plate_str = self.gen_plate_string(-1, -1)
            img = cv2.resize(self.generate(plate_str), size)
            cv2.imwrite(output_path + "/" + str(i).zfill(2) + ".jpg", img)

#
# g = GenPlate("platech.ttf", "platechar.ttf", "../env_images")
# g.gen_batch(15, 2, range(31, 65), "../plates", (272, 72))
