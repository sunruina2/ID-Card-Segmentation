from PIL import Image
import numpy as np
import cv2
import os, shutil
import random


def pil_cv2(image):
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def cv2_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def brenner(img_i):
    img_i = np.asarray(img_i, dtype='float64')
    x, y = img_i.shape
    img_i -= 127.5
    img_i *= 0.0078125  # 标准化
    center = img_i[0:x - 2, 0:y - 2]
    center_xplus = img_i[2:, 0:y - 2]
    center_yplus = img_i[0:x - 2:, 2:]
    Dx = int(np.sum((center_xplus - center) ** 2))
    Dy = int(np.sum((center_yplus - center) ** 2))
    return Dx, Dy


# 均值模糊 去随机噪声有很好的去燥效果
def blur_demo(image, k):
    print(k)
    dst = cv2.blur(image, (k, k))
    return dst


# 中值模糊去除椒盐噪声
def median_blur_demo(image):
    dst = cv2.medianBlur(image, 3)
    return dst


# 自定义模糊
def custom_blur_demo(image):
    kernels = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    dst = cv2.filter2D(image, -1, kernel=kernels)
    return dst


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
    # all_crop_pt = '/Users/finup/Desktop/图像语义分割/all_crop_id'
    # all_crop_pt_trans = all_crop_pt + '_trans'
    # try:
    #     os.mkdir(all_crop_pt_trans)
    # except:
    #     pass
    #
    # crop_lst = sorted(os.listdir(all_crop_pt))
    # all_n = len(crop_lst)
    # for i in range(all_n):
    #     if crop_lst[i].split('.')[-1] != '.DS_Store':
    #         if i % 10 == 0:
    #             print(all_n, i, np.round(i / all_n, 2))
    #         # 先确定图片的四个顶点的坐标：
    #         crop_img = cv2.imread(all_crop_pt + '/' + crop_lst[i])
    #         h, w = crop_img.shape[:2]
    #         pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]])
    #         # 逆时针 4点
    #         # [0, 0] 左上
    #         # [0, h - 1] 左下
    #         # [w - 1, h - 1] 右下
    #         # [w - 1, 0]] 右上
    #         r8 = [random.randint(0, 50) for i in range(8)]
    #         pts1 = np.float32([[0 + r8[0], 0 + r8[1]], [0 + r8[2], h - 1 - r8[3]], [w - 1 - r8[4], h - 1 - r8[5]],
    #                            [w - 1 - r8[6], 0 + r8[7]]])
    #         M = cv2.getPerspectiveTransform(pts, pts1)
    #         dst = cv2.warpPerspective(crop_img, M, (w, h))
    #
    #         # for p_i in range(len(pts1)):
    #         #     cv2.circle(dst, (int(pts1[p_i][0]), int(pts1[p_i][1])), 3, (0, 0, 255), 0)
    #         pts1 = np.reshape(pts1, (8,))
    #         r8s = [str(int(i)) for i in pts1]
    #         r8s = '_'.join(r8s)
    #         cv2.imwrite(all_crop_pt_trans + '/' + crop_lst[i].replace('_crop.jpg', '@' + r8s + '.jpg'), dst)

    '''读取狒狒身份证crop'''
    # 前景：身份证
    ft_path = '/Users/finup/Desktop/图像语义分割/all_crop_id_trans'
    all_id_lst = sorted(os.listdir(ft_path))
    if '.DS_Store' in all_id_lst:
        all_id_lst.remove('.DS_Store')
    id_n = len(all_id_lst)
    # 背景：coco
    bk_path = '/Users/finup/Desktop/图像语义分割/train2017_s/'
    bk_list = sorted(os.listdir(bk_path))
    bk_n = len(bk_list)
    if '.DS_Store' in bk_list:
        bk_list.remove('.DS_Store')
    # 新建存储地址
    ft_path_new = ft_path + '_cat/'
    try:
        os.mkdir(ft_path_new)
    except:
        pass
    # 随机参数设置

    # 遍历每张id照片
    for id_i in range(id_n):
        # 读取一张前景
        id_img = cv2.imread(ft_path + '/' + all_id_lst[id_i])
        r8 = np.reshape(all_id_lst[id_i].split('@')[-1].split('.')[0].split('_'), (4, 2)).astype(np.float)
        print('\n\n')
        print(id_i, all_id_lst[id_i], r8)
        hid, wid, _ = id_img.shape
        blur1id, blur2id = brenner(np.asarray(id_img)[:, :, 0])

        # 迭代选取背景
        for try_i in range(10):
            # 随机抽取一张背景
            bk_i = random.randint(0, bk_n - 2)
            print(bk_path + bk_list[bk_i])
            bk_img = cv2.imread(bk_path + bk_list[bk_i])
            hbk, wbk, _ = bk_img.shape
            # 判定背景大小是大于idcard大小，不符合进行随机延展
            exstend_gap = random.randint(100, 500)  # 延伸大小
            if hid + exstend_gap < hbk and wid + exstend_gap < wbk:
                pass
            else:
                bk_img = cv2.resize(bk_img, (wid + exstend_gap, hid + exstend_gap))
                hbk, wbk, _ = bk_img.shape

            # 判断背景清晰度是否大于idcard，满足的话进行清晰度降低和粘贴
            blur1bk, blur2bk = brenner(np.asarray(bk_img)[:, :, 0])
            r_blur = int(blur1bk / blur1id)
            print('before_blur:', blur1id, blur1bk, r_blur)
            if r_blur > 2:
                bk_img = blur_demo(bk_img, k=int(r_blur / 2))  # 降低清晰度
                blur1bk, blur2bk = brenner(np.asarray(bk_img)[:, :, 0])  # 统计降低后的清晰度
                r_blur = int(blur1bk / blur1id)  # 检查前景背景清晰度差异
                print('after_blur:', blur1id, blur1bk, r_blur)

                bk_img_black = np.zeros(bk_img.shape).astype(float)
                # 开始粘贴
                id_copy, bk_img = cv2_pil(id_img.copy()), cv2_pil(bk_img)
                dx = random.randint(int(-exstend_gap / 2), int(exstend_gap / 2))
                dy = random.randint(int(-exstend_gap / 2), int(exstend_gap / 2))
                print(dx, dy)
                left_up = (int((hbk - hid) / 2) + dy, int((wbk - wid) / 2) + dx)
                # right_down = (int((hbk - hid) / 2) + dy + wid, int((wbk - wid) / 2) + dx + hid)
                bk_img.paste(id_copy, left_up)
                bk_img = pil_cv2(bk_img)
                r8_trans = np.zeros(r8.shape)
                r8_trans = [[r8[i][0] + left_up[0], r8[i][1] + left_up[1]] for i in range(len(r8_trans))]
                # for p_i in range(len(r8_trans)):
                #     cv2.circle(bk_img, (int(r8_trans[p_i][0]), int(r8_trans[p_i][1])), 3, (0, 255, 0), 0)
                cv2.imwrite(ft_path_new + all_id_lst[id_i], bk_img)
                break
            else:
                continue
