import math
import os
import shutil
import cv2
import json
import numpy as np
from glob import glob

data_pt = '/Users/finup/Desktop/图像语义分割/'
data_pt = '/data/sunruina/img_segment/'


def read_image(img, label):
    image = cv2.imread(img)  # <class 'tuple'>: (1920, 1080, 3)

    mask = np.zeros(image.shape, dtype=np.uint8)  # <class 'tuple'>: (1920, 1080, 3)
    # cv2.imwrite(img.replace('.jpg', '_0mask.jpg'), mask)
    # print(mask.shape)

    # quad = json.load(open(label, 'r'))  # <class 'dict'>: {'quad': [[77, 640], [1004, 624], [1019, 1307], [32, 1281]]}
    quad = label
    coords = np.array(quad['quad'], dtype=np.int32)
    # [[  77  640] 左上
    #  [1004  624] 右上
    #  [1019 1307] 右下
    #  [  32 1281]] 左下
    cv2.fillPoly(mask, coords.reshape(-1, 4, 2), color=(255, 255, 255))  # 扩维
    # cv2.imwrite(img.replace('.jpg', '_1maskfill.jpg'), mask)
    # print(mask.shape)

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)  # mask由3通道变灰色单通道
    # cv2.imwrite(img.replace('.jpg', '_2maskfillgray.jpg'), mask)
    # print(mask.shape)

    mask = cv2.resize(mask, (mask.shape[1] // 2, mask.shape[0] // 2))
    # cv2.imwrite(img.replace('.jpg', '_3maskfillgraysize.jpg'), mask)
    # print(mask.shape)

    image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
    # cv2.imwrite(img.replace('.jpg', '_4imagesize.jpg'), image)
    # print(image.shape)

    # 二值化resize之后边缘会变成灰色，这一步是将边缘变为黑白，https://blog.csdn.net/sinat_21258931/article/details/61418681
    mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]
    # cv2.imwrite(img.replace('.jpg', '_5maskfillgraysizehold.jpg'), mask)
    # print(mask.shape)

    mask = cv2.resize(mask, (256, 256))
    # cv2.imwrite(img.replace('.jpg', '_5maskfillgraysizeholdsize.jpg'), mask)
    # print(mask.shape)

    image = cv2.resize(image, (256, 256))
    # cv2.imwrite(img.replace('.jpg', '_4imagesizesize.jpg'), image)
    # print(image.shape)

    return image, mask


def main():
    all_pic_path = data_pt + 'all_crop_id_trans_paste/'
    npy_patht = all_pic_path.replace('all_crop_id_trans_paste', 'all_crop_id_trans_paste_npyt')
    npy_pathm = all_pic_path.replace('all_crop_id_trans_paste', 'all_crop_id_trans_paste_npym')
    try:
        os.mkdir(npy_patht)
    except:
        pass
    try:
        os.mkdir(npy_pathm)
    except:
        pass
    save_n = 5000
    print_n = 100
    train_pic = sorted(os.listdir(all_pic_path))
    pic_n = len(train_pic)
    npy_n = math.floor(pic_n / float(save_n))
    for pic_i in range(pic_n):
        if pic_i == 0:
            X = []
            Y = []

        if pic_i % print_n == 0 and pic_i != 0:
            print(pic_n, pic_i, np.round(pic_i % pic_n, 2))

        if train_pic[pic_i] != '.DS_Store' and 'mask' not in train_pic[pic_i] and 'image' not in train_pic[pic_i]:
            img = all_pic_path + train_pic[pic_i]
            loc = np.asarray(train_pic[pic_i].split('@')[-1].split('.')[0].split('_'), dtype=int)
            label = {'quad': [[loc[0], loc[1]], [loc[6], loc[7]], [loc[4], loc[5]], [loc[2], loc[3]]]}
            image, mask = read_image(img, label)  # <class 'tuple'>: (256, 256, 3)，<class 'tuple'>: (256, 256)
            X.append(image)
            Y.append(mask)
            if (pic_i != 0 and pic_i % save_n == 0) or (pic_i != npy_n * save_n - 1 and pic_i == pic_n - 1):
                print(pic_i, 'save..', len(X))
                X = np.array(X)
                Y = np.array(Y)
                Y = np.expand_dims(Y, axis=3)
                np.save(npy_patht + 'train_image' + str(pic_i) + '.npy', X)
                np.save(npy_pathm + 'mask_image' + str(pic_i) + '.npy', Y)
                X = []
                Y = []


def main_concat_npy():
    npy_patht = data_pt + 'all_crop_id_trans_paste_npyt/'
    npy_pathm = data_pt + 'all_crop_id_trans_paste_npym/'
    npy_path_tm = data_pt + 'all_crop_id_trans_paste_npy/'

    npy_path_filest = sorted(os.listdir(npy_patht))
    npy_path_filesm = sorted(os.listdir(npy_pathm))
    try:
        os.mkdir(npy_path_tm)
    except:
        pass

    if '.DS_Store' in npy_path_filest:
        npy_path_filest.remove('.DS_Store')
    if '.DS_Store' in npy_path_filesm:
        npy_path_filesm.remove('.DS_Store')

    for f in range(len(npy_path_filest)):
        if f == 0:
            total = np.load(npy_patht + npy_path_filest[f])
        else:
            temp = np.load(npy_patht + npy_path_filest[f])
            total = np.vstack((total, temp))  # (120, 256, 256, 3)
    print('concat x train', total.shape)
    np.save(npy_path_tm + 'final_train.npy', total)

    for f in range(len(npy_path_filesm)):
        if f == 0:
            total = np.load(npy_pathm + npy_path_filesm[f])
        else:
            temp = np.load(npy_pathm + npy_path_filesm[f])
            total = np.vstack((total, temp))  # (120, 256, 256, 1)
    print('concat y mask', total.shape)
    np.save(npy_path_tm + 'final_mask.npy', total)


if __name__ == '__main__':
    main()
    main_concat_npy()
