"""
车牌定位

"""

import cv2
import numpy as np

"""
# 图像拉伸函数的原理

    图像拉伸是为了让原来图像（灰度图）的像素点灰度全部落到指定范围内。
    该函数的主要功能通过for循环实现。
    像素拉伸的公式如下：
    
    new_value = (255 / (max_value - min_value)) * (original_value - min_value)
    
    其中，`(255 / (max_value - min_value))`用于计算缩放因子，
    该因子可以将原始像素值的范围映射到 0 到 255 的范围。这样做是为了确保最亮的像素值对应于 255，最暗的像素值对应于 0。
    
    将原始像素值减去最小值，得到一个相对于最小值的偏移量。
    然后将这个偏移量乘以缩放因子，得到新的像素值。这个新的像素值就是经过拉伸计算后的结果。

"""


def stretch(img):
    """
    图像拉伸函数，
    通过对每个像素点进行调整，将所有像素点的灰度限制到0-255之间。

    :param img: 图像数据，可以是一个二维数组
    :return: 经过拉伸后的图像数据
    """
    maxi = float(img.max())  # 获取图像的最大和最小像素值
    mini = float(img.min())

    # 循环遍历每个像素点，拉伸像素
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = (255 / (maxi - mini) * img[i, j] - (255 * mini) / (maxi - mini))  # 图像拉伸函数

    return img


def dobinarization(img):
    """
    二值化处理函数
    首先计算出阈值x，然后通过与x比对并修改像素值，实现二值化。

    :param img: 被二值化处理的图像
    :return thresh: 返回二值化后的图像
    """
    maxi = float(img.max())  # 获取最大最小像素值，转化成float类型
    mini = float(img.min())

    # 计算阈值 x
    x = maxi - ((maxi - mini) / 2)  # 通过计算中值，可以将像素值均分到大于阈值和小于等于阈值两个部分
    # 二值化,返回阈值ret  和  二值化操作后的图像thresh
    # 原始图像、阈值、输出的最大像素值、阈值化类型
    ret, thresh = cv2.threshold(img, x, 255, cv2.THRESH_BINARY)
    # 返回二值化后的黑白图像
    return thresh


def find_rectangle(contour):
    """
    寻找矩形轮廓

    :param contour: 轮廓的点集
    :return list(): 返回轮廓的左上角和右下角坐标
    """
    y, x = [], []  # 定义列表

    for p in contour:  # 遍历轮廓集上的每一个点
        y.append(p[0][0])  # 点的x坐标
        x.append(p[0][1])  # 点的y坐标

    return [min(y), min(x), max(y), max(x)]  # 返回轮廓的左上角和右下角坐标


def locate_license(img, afterimg):
    """
    定位车牌号

    :param img: 输入的二值化图像
    :param afterimg:
    :return:
    """

    # 找到所有轮廓
    # RETR_EXTERNAL 表示只找到最外层的图像；CHAIN_APPROX_SIMPLE 表示使用简单的轮廓逼近法存储轮廓点
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 找出最大的三个区域
    block = []
    for c in contours:
        # 找出轮廓的左上点和右下点，由此计算它的面积和长度比
        r = find_rectangle(c)  # 轮廓的对角点坐标
        a = (r[2] - r[0]) * (r[3] - r[1])  # 面积
        s = (r[2] - r[0]) * (r[3] - r[1])  # 长度比

        block.append([r, a, s])
    # 选出 面积最大 的3个区域
    block = sorted(block, key=lambda b: b[1])[-3:]

    # 使用颜色识别判断找出最像车牌的区域
    maxweight, maxindex = 0, -1
    for i in range(len(block)):  # 遍历筛选后的三个最大区域
        b = afterimg[block[i][0][1]:block[i][0][3], block[i][0][0]:block[i][0][2]]
        # BGR转HSV
        hsv = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
        # 蓝色车牌的范围
        lower = np.array([100, 50, 50])
        upper = np.array([140, 255, 255])
        # 根据阈值构建掩膜
        mask = cv2.inRange(hsv, lower, upper)
        # 统计权值
        w1 = 0
        for m in mask:
            w1 += m / 255

        w2 = 0
        for n in w1:
            w2 += n

        # 选出最大权值的区域
        if w2 > maxweight:
            maxindex = i
            maxweight = w2

    return block[maxindex][0]
