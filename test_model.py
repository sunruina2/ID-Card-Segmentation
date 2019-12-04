import cv2
import numpy as np
from keras.models import load_model
from keras.optimizers import Adam
from model.iou_loss import IoU
import matplotlib.pyplot as plt

model = load_model('unet_model_whole_100epochs.h5', compile=False)
model.compile(optimizer=Adam(1e-4), loss=IoU, metrics=['binary_accuracy'])

image_name = '2'
pro_path = '/Users/finup/Desktop/图像语义分割/ID-Card-Segmentation/images/'
img = cv2.imread(pro_path + image_name + '.jpg')

h, w = img.shape[:2]
img = cv2.resize(img, (256, 256))
img = img / 255.0
predict = model.predict(img.reshape(1, 256, 256, 3))

output = predict[0]
output = cv2.resize(output, (w, h))
plt.imsave(pro_path + image_name + '_out.jpg', output, cmap='gray')
