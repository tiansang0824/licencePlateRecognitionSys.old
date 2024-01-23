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
    :param afterimg: 用于标记车牌区域的图片
    :return block[maxindex][0]: 车牌区域
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
        # lower和upper是HSV格式下蓝色的范围
        lower = np.array([100, 50, 50])
        upper = np.array([140, 255, 255])
        # 根据阈值构建掩膜
        # cv2.inRange() 负责根据指定的颜色范围构建掩膜图像。
        # 掩膜图像是一个二值化的图像，在颜色范围内的像素值为255（白色），颜色范围外的像素值为0（黑色）。
        # mask是一个二值图像。
        mask = cv2.inRange(hsv, lower, upper)

        # 统计权值
        w1 = 0
        for m in mask:  # 计算掩膜中像素值为255的像素的数量。
            w1 += m / 255

        w2 = 0
        for n in w1:  # 将w1中的所有权值相加，得到最终权值。
            w2 += n

        # 选出最大权值的区域
        if w2 > maxweight:
            maxindex = i
            maxweight = w2

    return block[maxindex][0]


def find_license(img):
    """
    预处理函数

    :param img: 传入原图
    :return rect: 车牌位置信息，该信息通过调用locate_licence()实现
    :return img: 返回被重置尺寸后的原图
    """
    """
    第一步、压缩图像
    由于要把宽度压缩为400px，所以要先利用原图的比例计算压缩后的图像的高度（height/rows）
    """
    m = 400 * img.shape[0] / img.shape[1]
    # 压缩图像
    img = cv2.resize(img, (400, int(m)), interpolation=cv2.INTER_CUBIC)

    # BGR转换为灰度图像
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 灰度拉伸
    # 这是前面定义过的函数，用于将灰度图的灰度限定到指定范围（0,255）内
    stretched_img = stretch(gray_img)
    """
    开运算去除噪声
    
    - 开运算一般可用于去除图片中的细小物体、在纤细点分离，在平滑较大物体的边界的同时并不明显改变其面积。
    - 闭运算可以用于填充物体内部的小空洞、连接临近物体、并且平滑其边界。
    
    """
    r = 16  # 设置卷积核的半径
    h = w = r * 2 + 1  # 卷积核的宽度和高度，目前kernel的大小是33x33
    kernel = np.zeros((h, w), np.uint8)  # 创建卷积核（一个全0数组）
    # 在全零数组的基础上，使用cv2.circle函数在这个kernel上画一个填充的圆。
    # 创造出一个以r为半径的实心圆形的kernel，用于后续的开运算。
    cv2.circle(kernel, (r, r), r, 1, -1)
    # 开运算
    # 开运算参数：拉伸后的图像、开运算标识符、开运算卷积核
    opening_img = cv2.morphologyEx(stretched_img, cv2.MORPH_OPEN, kernel)
    # 获取差分图，两幅图像做差  cv2.absdiff('图像1','图像2')
    # 计算拉伸后的图像和开运算后的图像的差分，这有助于突出显示图像中的边缘，这些边缘可能是车牌的边缘。
    strt_img = cv2.absdiff(stretched_img, opening_img)

    # 图像二值化
    binary_img = dobinarization(strt_img)
    # canny边缘检测
    canny = cv2.Canny(binary_img, binary_img.shape[0], binary_img.shape[1])

    """
    消除小的区域，保留大块的区域，从而定位车牌
    """
    # 进行闭运算
    kernel = np.ones((5, 19), np.uint8)  # 创建卷积核
    closing_img = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)  # 闭运算
    # 进行开运算
    opening_img = cv2.morphologyEx(closing_img, cv2.MORPH_OPEN, kernel)
    # 再次进行开运算
    kernel = np.ones((11, 5), np.uint8)
    opening_img = cv2.morphologyEx(opening_img, cv2.MORPH_OPEN, kernel)

    # 消除小区域，定位车牌位置
    # rect是车牌位置信息、opening_img是开闭运算处理后的结果、img是用来截图区域的原图
    rect = locate_license(opening_img, img)
    print(f'rect = {rect}')

    return rect, img


def cut_license(after_img, rect):
    """
    图像分割函数
    :param after_img: 预处理后的图像
    :param rect:
    :return img_show: 分割后的图像
    """
    """
    rect 来自locate_license()，是某一个轮廓经过find_rectangle()函数处理后的矩形轮廓，
    rect 是一个列表：[最小x坐标，最小y坐标，最大x坐标，最大y坐标]
    """
    # 转换为宽度和高度
    rect[2] = rect[2] - rect[0]  # 将第三个元素值转换为宽度
    rect[3] = rect[3] - rect[1]  # 将第四个元素值转换为高度
    rect_copy = tuple(rect.copy())  # 转换成元组数据
    rect = [0, 0, 0, 0]
    # 创建掩膜
    mask = np.zeros(after_img.shape[:2], np.uint8)  # 通过预处理后的图像创建掩膜
    # 创建背景模型  大小只能为13*5，行数只能为1，单通道浮点型
    bgdModel = np.zeros((1, 65), np.float64)
    # 创建前景模型
    fgdModel = np.zeros((1, 65), np.float64)
    # 分割图像
    cv2.grabCut(after_img, mask, rect_copy, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img_show = after_img * mask2[:, :, np.newaxis]

    return img_show


def deal_license(licenseimg):
    """
    车牌图片二值化

    :param licenseimg: 车牌图片
    :return thresh: 二值化后的图片
    """
    # 车牌变为灰度图像
    gray_img = cv2.cvtColor(licenseimg, cv2.COLOR_BGR2GRAY)

    # 均值滤波  去除噪声
    kernel = np.ones((3, 3), np.float32) / 9  # 创建卷积核
    gray_img = cv2.filter2D(gray_img, -1, kernel)  # 进行均值滤波

    # 二值化处理
    ret, thresh = cv2.threshold(gray_img, 120, 255, cv2.THRESH_BINARY)

    return thresh


if __name__ == '__main__':

    """
    下面的算法只能定位到水平车牌
    """

    img = cv2.imread('test.png', cv2.IMREAD_COLOR)
    # 预处理图像
    rect, afterimg = find_license(img)

    # 框出车牌号
    cv2.rectangle(afterimg, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
    cv2.imshow('afterimg', afterimg)

    # 分割车牌与背景
    cutimg = cut_license(afterimg, rect)
    cv2.imshow('cutimg', cutimg)

    # 二值化生成黑白图
    thresh = deal_license(cutimg)
    cv2.imshow('thresh', thresh)
    cv2.imwrite("cp.jpg", thresh)
    cv2.waitKey(0)
