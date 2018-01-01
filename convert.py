from PIL import Image
image_file = Image.open("./plates/00.jpg") # open colour image
image_file = image_file.convert('L') # convert image to black and white
image_file.save('./plates/15.jpg')

import cv2
import numpy as np

image1 = cv2.imread('./plates/00.jpg', cv2.IMREAD_GRAYSCALE)
# image1 = np.multiply()
image1 = np.multiply(image1, 1 / 255.0)
image1 = np.array(image1)
image2 = cv2.imread('./plates/15.jpg')
image2 = np.multiply(image2, 1 / 255.0)
image2 = np.array(image2)

print(image1)
print(image2)

