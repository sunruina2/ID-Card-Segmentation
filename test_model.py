import cv2
import numpy as np
from keras.models import load_model
from keras.optimizers import Adam
from model.iou_loss import IoU
import os
import time


def min_rect(p_name, img, img_origin):
    ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
    contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        # find minimum area
        rect = cv2.minAreaRect(c)
        # calculate coordinates of the minimum area rectangle
        box = cv2.boxPoints(rect)
        # normalize coordinates to integers
        box = np.int0(box)  # 左下角开始，顺时针
        img_origin_draw = img_origin.copy()
        cv2.drawContours(img_origin_draw, [box], 0, (0, 0, 255), 2)
        for p_i in range(len(box)):
            cv2.circle(img_origin_draw, (int(box[p_i][0]), int(box[p_i][1])), 3, (0, 255, 0), 1)
        box = np.reshape(box, (8,))
        r8s = [str(int(i)) for i in box]
        r8s = '_'.join(r8s)
        cv2.imwrite(pic_res_path + p_name + '_res@' + r8s + '.jpg', img_origin_draw)

        # 旋转
        width = int(rect[1][0])
        height = int(rect[1][1])
        angle = rect[2]
        if width < height:  # 计算角度，为后续做准备
            angle = angle - 90
        src_pts = cv2.boxPoints(rect)
        dst_pts = np.array([[0, height],
                            [0, 0],
                            [width, 0],
                            [width, height]], dtype="float32")
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(img_origin, M, (width, height))

        if angle <= -90:  # 对-90度以上图片的竖直结果转正
            warped = cv2.transpose(warped)
            warped = cv2.flip(warped, 0)  # 逆时针转90度，如果想顺时针，则0改为1

        cv2.imwrite(pic_res_path + p_name + '_res.jpg', warped)


class UNetIdCard():
    def __init__(self):
        self.epoch_n = '_27'
        self.model = load_model('unet_model_whole_100epochs' + self.epoch_n + '.h5', compile=False)
        self.model.compile(optimizer=Adam(1e-4), loss=IoU, metrics=['binary_accuracy'])

    def pre(self, image_name):
        img_origin = cv2.imread(pic_path + image_name + '.jpg')
        img = img_origin.copy()
        h, w = img.shape[:2]
        img = cv2.resize(img, (256, 256))
        img = img / 255.0
        predict = self.model.predict(img.reshape(1, 256, 256, 3))
        output = predict[0]
        output = cv2.resize(output, (w, h))

        # 删除小的连通区域
        b = np.where(output > 0.5, 255, 0).astype(np.uint8)
        mask = cv2.merge([b, b, b])
        cv2.imwrite(pic_res_path + image_name + '_mask_all.jpg', mask)
        _, binary = cv2.threshold(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY), 0.1, 1, cv2.THRESH_BINARY)
        contours, hierarch = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area < 0.2 * w * h:
                cv2.drawContours(mask, [contours[i]], 0, 0, -1)
        cv2.imwrite(pic_res_path + image_name + '_mask_max.jpg', mask)

        min_rect(image_name, mask, img_origin)


if __name__ == '__main__':
    m = UNetIdCard()
    pic_path = '/Users/finup/Desktop/图像语义分割/ID-Card-Segmentation/test_imgs/'
    pic_res_path = pic_path + 'imgs_res/'
    try:
        os.mkdir(pic_res_path)
    except:
        pass

    pic_list = sorted(os.listdir(pic_path))
    pic_names = []
    for i in pic_list:
        name_s = i.split('.')
        if name_s[-1] == 'jpg':
            pic_names.append(name_s[0])

    for i in pic_names:
        start = time.time()
        m.pre(i)
        end = time.time()
        print(end - start)
