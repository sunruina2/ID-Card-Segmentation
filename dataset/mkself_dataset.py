'''
This script just downloads the dataset and stores the both images and their respective masks
in numpy file. There will be 50 .npy files, you can use them seperately or can run other script
to combine all of them in single file.
'''

import os
import shutil
import cv2
import json
import numpy as np
from glob import glob

all_links = ['ftp://smartengines.com/midv-500/dataset/01_alb_id.zip',
             'ftp://smartengines.com/midv-500/dataset/02_aut_drvlic_new.zip',
             'ftp://smartengines.com/midv-500/dataset/03_aut_id_old.zip',
             'ftp://smartengines.com/midv-500/dataset/04_aut_id.zip',
             'ftp://smartengines.com/midv-500/dataset/05_aze_passport.zip',
             'ftp://smartengines.com/midv-500/dataset/06_bra_passport.zip',
             'ftp://smartengines.com/midv-500/dataset/07_chl_id.zip',
             'ftp://smartengines.com/midv-500/dataset/08_chn_homereturn.zip',
             'ftp://smartengines.com/midv-500/dataset/09_chn_id.zip',
             'ftp://smartengines.com/midv-500/dataset/10_cze_id.zip',
             'ftp://smartengines.com/midv-500/dataset/11_cze_passport.zip',
             'ftp://smartengines.com/midv-500/dataset/12_deu_drvlic_new.zip',
             'ftp://smartengines.com/midv-500/dataset/13_deu_drvlic_old.zip',
             'ftp://smartengines.com/midv-500/dataset/14_deu_id_new.zip',
             'ftp://smartengines.com/midv-500/dataset/15_deu_id_old.zip',
             'ftp://smartengines.com/midv-500/dataset/16_deu_passport_new.zip',
             'ftp://smartengines.com/midv-500/dataset/17_deu_passport_old.zip',
             'ftp://smartengines.com/midv-500/dataset/18_dza_passport.zip',
             'ftp://smartengines.com/midv-500/dataset/19_esp_drvlic.zip',
             'ftp://smartengines.com/midv-500/dataset/20_esp_id_new.zip',
             'ftp://smartengines.com/midv-500/dataset/21_esp_id_old.zip',
             'ftp://smartengines.com/midv-500/dataset/22_est_id.zip',
             'ftp://smartengines.com/midv-500/dataset/23_fin_drvlic.zip',
             'ftp://smartengines.com/midv-500/dataset/24_fin_id.zip',
             'ftp://smartengines.com/midv-500/dataset/25_grc_passport.zip',
             'ftp://smartengines.com/midv-500/dataset/26_hrv_drvlic.zip',
             'ftp://smartengines.com/midv-500/dataset/27_hrv_passport.zip',
             'ftp://smartengines.com/midv-500/dataset/28_hun_passport.zip',
             'ftp://smartengines.com/midv-500/dataset/29_irn_drvlic.zip',
             'ftp://smartengines.com/midv-500/dataset/30_ita_drvlic.zip',
             'ftp://smartengines.com/midv-500/dataset/31_jpn_drvlic.zip',
             'ftp://smartengines.com/midv-500/dataset/32_lva_passport.zip',
             'ftp://smartengines.com/midv-500/dataset/33_mac_id.zip',
             'ftp://smartengines.com/midv-500/dataset/34_mda_passport.zip',
             'ftp://smartengines.com/midv-500/dataset/35_nor_drvlic.zip',
             'ftp://smartengines.com/midv-500/dataset/36_pol_drvlic.zip',
             'ftp://smartengines.com/midv-500/dataset/37_prt_id.zip',
             'ftp://smartengines.com/midv-500/dataset/38_rou_drvlic.zip',
             'ftp://smartengines.com/midv-500/dataset/39_rus_internalpassport.zip',
             'ftp://smartengines.com/midv-500/dataset/40_srb_id.zip',
             'ftp://smartengines.com/midv-500/dataset/41_srb_passport.zip',
             'ftp://smartengines.com/midv-500/dataset/42_svk_id.zip',
             'ftp://smartengines.com/midv-500/dataset/43_tur_id.zip',
             'ftp://smartengines.com/midv-500/dataset/44_ukr_id.zip',
             'ftp://smartengines.com/midv-500/dataset/45_ukr_passport.zip',
             'ftp://smartengines.com/midv-500/dataset/46_ury_passport.zip',
             'ftp://smartengines.com/midv-500/dataset/47_usa_bordercrossing.zip',
             'ftp://smartengines.com/midv-500/dataset/48_usa_passportcard.zip',
             'ftp://smartengines.com/midv-500/dataset/49_usa_ssn82.zip',
             'ftp://smartengines.com/midv-500/dataset/50_xpo_id.zip']

all_links = ['ftp://smartengines.com/midv-500/dataset/06_bra_passport.zip',
             'ftp://smartengines.com/midv-500/dataset/15_deu_id_old.zip'

             ]


def read_image(img, label):
    print('\n\n')
    print(img)
    image = cv2.imread(img)  # <class 'tuple'>: (1920, 1080, 3)
    cv2.imwrite(img.replace('.tif', '.jpg'), image)

    mask = np.zeros(image.shape, dtype=np.uint8)  # <class 'tuple'>: (1920, 1080, 3)
    cv2.imwrite(img.replace('.tif', '_0mask.jpg'), mask)
    print(mask.shape)

    quad = json.load(open(label, 'r'))  # <class 'dict'>: {'quad': [[77, 640], [1004, 624], [1019, 1307], [32, 1281]]}
    coords = np.array(quad['quad'], dtype=np.int32)
    # [[  77  640] 左上
    #  [1004  624] 右上
    #  [1019 1307] 右下
    #  [  32 1281]] 左下
    cv2.fillPoly(mask, coords.reshape(-1, 4, 2), color=(255, 255, 255))  # 扩维
    # [[[  77  640]
    #   [1004  624]
    #   [1019 1307]
    #   [  32 1281]]]
    cv2.imwrite(img.replace('.tif', '_1maskfill.jpg'), mask)
    print(mask.shape)

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)  # mask由3通道变灰色单通道
    cv2.imwrite(img.replace('.tif', '_2maskfillgray.jpg'), mask)
    print(mask.shape)

    mask = cv2.resize(mask, (mask.shape[1] // 2, mask.shape[0] // 2))
    cv2.imwrite(img.replace('.tif', '_3maskfillgraysize.jpg'), mask)
    print(mask.shape)

    image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
    cv2.imwrite(img.replace('.tif', '_4imagesize.jpg'), image)
    print(image.shape)
    # 二值化resize之后边缘会变成灰色，这一步是将边缘变为黑白，https://blog.csdn.net/sinat_21258931/article/details/61418681
    mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite(img.replace('.tif', '_5maskfillgraysizehold.jpg'), mask)
    print(mask.shape)

    mask = cv2.resize(mask, (256, 256))
    cv2.imwrite(img.replace('.tif', '_5maskfillgraysizeholdsize.jpg'), mask)
    print(mask.shape)

    image = cv2.resize(image, (256, 256))
    cv2.imwrite(img.replace('.tif', '_4imagesizesize.jpg'), image)
    print(image.shape)

    return image, mask


def main():
    i = 1
    all_pic = '/Users/finup/Desktop/图像语义分割/all_crop_id_trans_paste'
    train_pic = os.listdir(all_pic)
    pic_n = len(train_pic)


    for pic_i in range(pic_n):

    for img, label in zip(img_list, label_list):
         image, mask = read_image(img, label)  # <class 'tuple'>: (256, 256, 3)，<class 'tuple'>: (256, 256)
         X.append(image)
         Y.append(mask)

    X = np.array(X)
    Y = np.array(Y)
    Y = np.expand_dims(Y, axis=3)
    print(link[40:].replace('.zip', ''), X.shape, Y.shape)
    np.save('train_image' + str(i) + '.npy', X)
    np.save('mask_image' + str(i) + '.npy', Y)
    print('Files Saved For:', link[40:].replace('.zip', ''))
    i += 1
    print('----------------------------------------------------------------------')
    # os.remove(link[40:])
    # shutil.rmtree(link[40:].replace('.zip', ''))


if __name__ == '__main__':
    main()
