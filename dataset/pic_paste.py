from PIL import Image
import numpy as np
import cv2
import os, shutil
import random

# def pil_cv2(image):
#     return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
#
#
# def cv2_pil(img):
#     return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#
#
# def brenner(img_i):
#     img_i = np.asarray(img_i, dtype='float64')
#     x, y = img_i.shape
#     img_i -= 127.5
#     img_i *= 0.0078125  # 标准化
#     center = img_i[0:x - 2, 0:y - 2]
#     center_xplus = img_i[2:, 0:y - 2]
#     center_yplus = img_i[0:x - 2:, 2:]
#     Dx = int(np.sum((center_xplus - center) ** 2))
#     Dy = int(np.sum((center_yplus - center) ** 2))
#     return Dx, Dy
#
#
# # 均值模糊 去随机噪声有很好的去燥效果
# def blur_demo(image, k):
#     print(k)
#     dst = cv2.blur(image, (k, k))
#     return dst
#
#
# # 中值模糊去除椒盐噪声
# def median_blur_demo(image):
#     dst = cv2.medianBlur(image, 3)
#     return dst
#
#
# # 自定义模糊
# def custom_blur_demo(image):
#     kernels = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
#     dst = cv2.filter2D(image, -1, kernel=kernels)
#     return dst


if __name__ == "__main__":

    '''查看狒狒身份证大小，原图不一致，crop图都是(560, 880, 3)'''
    # # all_crop_pt = '/Users/finup/Desktop/图像语义分割/all_crop_id/'
    # all_crop_pt = '/data/sunruina/img_segment/all_crop_id/'
    # try:
    #     os.mkdir(all_crop_pt)
    # except:
    #     pass
    # # id_rts_path = '/Users/finup/Desktop/图像语义分割/image_s/'
    # id_rts_path = '/data/sunruina/mfs/ff_id_card/image/'
    # id_rts_path_lst = sorted(os.listdir(id_rts_path))
    # i = 0
    # for rt in id_rts_path_lst:
    #     if rt.split('.')[-1] != 'DS_Store':
    #         print(rt)
    #         id_roots_path = id_rts_path+'/' + rt + '/'
    #         id_peo_path = sorted(os.listdir(id_roots_path))
    #         size_lst = []
    #         for peo in id_peo_path:
    #             if peo.split('.')[-1] != 'DS_Store':
    #                 id_pic_path = sorted(os.listdir(id_roots_path + peo))
    #                 for pic in id_pic_path:
    #                     if pic.split('_')[-1] == 'crop.jpg':  # crop结果同尺寸(560, 880, 3)
    #                         print(i)
    #                         i += 1
    #                         srcfile, dstfile = id_roots_path + peo + '/' + pic, all_crop_pt + str(i) + pic
    #                         shutil.copyfile(srcfile, dstfile)

    '''将身份证原图进行一定范围的透视变换，输入输出大小不变(560, 880, 3)，保存变换后的点，留白置为黑色'''

    all_crop_pt = '/Users/finup/Desktop/图像语义分割/all_crop_id'
    # all_crop_pt = '/data/sunruina/img_segment/all_crop_id'
    all_crop_pt_trans = all_crop_pt + '_trans'
    try:
        os.mkdir(all_crop_pt_trans)
    except:
        pass

    crop_lst = sorted(os.listdir(all_crop_pt))
    all_n = len(crop_lst)
    for i in range(all_n):
        if crop_lst[i].split('.')[-1] != '.DS_Store':
            if i % 10 == 0:
                print(all_n, i, np.round(i / all_n, 2))
            # 先确定图片的四个顶点的坐标：
            crop_img = cv2.imread(all_crop_pt + '/' + crop_lst[i])
            h, w = crop_img.shape[:2]
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]])
            # 逆时针 4点
            # [0, 0] 左上
            # [0, h - 1] 左下
            # [w - 1, h - 1] 右下
            # [w - 1, 0]] 右上
            r8 = [random.randint(0, 50) for i in range(8)]
            r8s = [str(i) for i in r8]
            r8s = '_'.join(r8s)
            pts1 = np.float32([[0 + r8[0], 0 + r8[1]], [0 + r8[2], h - 1 - r8[3]], [w - 1 - r8[4], h - 1 - r8[5]],
                               [w - 1 - r8[6], 0 + r8[7]]])
            M = cv2.getPerspectiveTransform(pts, pts1)
            dst = cv2.warpPerspective(crop_img, M, (w, h))
            cv2.imwrite(all_crop_pt_trans + '/' + crop_lst[i].replace('_crop.jpg', '@' + r8s + '.jpg'), dst)

    '''读取狒狒身份证crop'''
    # bk_path = '/Users/finup/Desktop/图像语义分割/train2017_s'
    # bk_path_dst = bk_path + '_bkres/'
    # bk_path = bk_path + '/'
    # exstend_gap = random.randint(100, 500)
    #
    # try:
    #     os.mkdir(bk_path_dst)
    # except:
    #     pass
    #
    # bk_list = sorted(os.listdir(bk_path))
    #
    # id_img = cv2.imread("/Users/finup/Desktop/图像语义分割/front_crop.jpg")
    # hid, wid, _ = id_img.shape
    # blur1id, blur2id = brenner(np.asarray(id_img)[:, :, 0])
    #
    # for file in bk_list:
    #     if file.split('.')[-1] == 'jpg':
    #         print('\n\n')
    #         print(file)
    #         bk_img = cv2.imread(bk_path + file)
    #         hbk, wbk, _ = bk_img.shape
    #
    #         if hid + exstend_gap < hbk and wid + exstend_gap < wbk:
    #             pass
    #         else:
    #             bk_img = cv2.resize(bk_img, (wid + exstend_gap, hid + exstend_gap))
    #             hbk, wbk, _ = bk_img.shape
    #
    #         blur1bk, blur2bk = brenner(np.asarray(bk_img)[:, :, 0])
    #         r_blur = int(blur1bk / blur1id)
    #         print('before_blur:', blur1id, blur1bk, r_blur)
    #         if r_blur > 2:
    #             bk_img = blur_demo(bk_img, k=int(r_blur / 2))
    #
    #             blur1bk, blur2bk = brenner(np.asarray(bk_img)[:, :, 0])
    #             r_blur = int(blur1bk / blur1id)
    #             print('after_blur:', blur1id, blur1bk, r_blur)
    #
    #             id_copy, bk_img = cv2_pil(id_img.copy()), cv2_pil(bk_img)
    #             dx = random.randint(int(-exstend_gap / 2), int(exstend_gap / 2))
    #             dy = random.randint(int(-exstend_gap / 2), int(exstend_gap / 2))
    #             print(dx, dy)
    #             bk_img.paste(id_copy, (int((hbk - hid) / 2) + dy,
    #                                    int((wbk - wid) / 2) + dx))
    #             bk_img = pil_cv2(bk_img)
    #             cv2.circle(bk_img, (int((hbk - hid) / 2) + dy,
    #                                 int((wbk - wid) / 2) + dx), 3, (0, 0, 255), 0)
    #             cv2.circle(bk_img, (int((hbk - hid) / 2) + dy + wid,
    #                                 int((wbk - wid) / 2) + dx + hid), 3, (0, 0, 255), 0)
    #             cv2.imwrite(bk_path_dst + 'bk_' + file, bk_img)
