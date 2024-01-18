"""
this model is used to locate the plates in the raw image.
after processing, it will return an image witch contains the area of the plate.
===== ===== ===== =====
这个模块用于在原始图像中定位车牌，在车牌定位处理后，将会返回一个只含有车牌区域的图片。
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from source.ToolKit import *


def locate_plate(raw_image):
    """
    用于车牌定位的函数，该函数会调用本模组中的其他函数，通过互相组合，实现车牌定位。
    :param raw_image: 含有车牌的原始图片
    :return: 返回车牌区域图片
    """
    pass


def pre_process(raw_image):
    """
    本函数用于对原始图片进行预处理，包括平滑、去噪、二值化、边缘检测、自适应阈值处理
    :param raw_image: 待处理的原始图片
    :return: 返回处理后的图片
    """
    pass


def gauss_process(raw_image):
    """
    本函数用于对图片进行高斯去噪
    :param raw_image: 原始图片
    :return: 返回高斯去噪后的结果
    """
    gauss_image = cv.GaussianBlur(raw_image, (3, 3), 0)
    return gauss_image


def grayscale_process(raw_image):
    """
    本函数用于对原始图像进行灰度处理
    :param raw_image:
    :return:
    """
    grayscale_outcome_image = cv.cvtColor(raw_image, cv.COLOR_BGR2GRAY)
    return grayscale_outcome_image


def edge_detect(raw_image):
    """
    对原图像进行边缘检测，返回检测结果。
    :param raw_image: 等待进行边缘检测的原图。
    :return: 返回一个图像，该图像为对原图进行边缘检测的结果。
    """
    sobel_x = cv.Sobel(raw_image, cv.CV_16S, 1, 0, ksize=3)
    abs_x = cv.convertScaleAbs(sobel_x)
    return abs_x


def adaptive_threshold(raw_image):
    """
    自适应阈值处理。
    :param raw_image:
    :return:
    """
    ret, adaptive_image = cv.threshold(raw_image, 0, 255, cv.THRESH_OTSU)
    return ret, adaptive_image


def closed_operation(raw_image):
    """
    对原图片进行闭运算，并且去除闭运算后图像中存在的白点
    :param raw_image: 原图
    :return: 返回闭运算后的图片
    """

    # 进行闭运算
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (14, 5))  # 创建卷积核
    closed_operated_image = cv.morphologyEx(raw_image, cv.MORPH_CLOSE, kernel, iterations=1)  # 进行闭运算

    # 去除白点
    # 创建卷积核
    kernel_x = cv.getStructuringElement(cv.MORPH_RECT, (20, 1))
    kernel_y = cv.getStructuringElement(cv.MORPH_RECT, (1, 19))
    # 膨胀、腐蚀
    image = cv.dilate(closed_operated_image, kernel_x)  # 膨胀
    image = cv.erode(image, kernel_x)  # 腐蚀
    # 腐蚀、膨胀
    image = cv.erode(image, kernel_y)
    image = cv.dilate(image, kernel_y)

    return image  # 返回处理后的图片


def median_filter(raw_image):
    """
    对原图进行中值滤波
    :param raw_image: 原图
    :return: 滤波结果
    """
    median_image = cv.medianBlur(raw_image, 15)
    return median_image


def detect_contours(img_for_contours, origin_image):
    """
    轮廓检测
    该函数会通过img_for_contours检测出所有轮廓，然后在origin_image中标出轮廓
    :param img_for_contours: 应传入中值滤波后的图片
    :param origin_image: 原图片，这个图片用于被标记轮廓，建议使用被处理的图片。
    :return:
    """
    # 轮廓检测
    contours, hierarchy = cv.findContours(img_for_contours, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # 绘制轮廓
    image_copy = origin_image.copy()
    cv.drawContours(image_copy, contours, -1, (0, 255, 0), 2)
    return contours, image_copy  # 返回被标记轮廓的图片复制品


def detect_contours_copy(img_for_contours, origin_image):
    """
    轮廓检测
    该函数会通过img_for_contours检测出所有轮廓，然后在origin_image中标出轮廓
    :param img_for_contours: 应传入中值滤波后的图片
    :param origin_image: 原图片，这个图片用于被标记轮廓，建议使用被处理的图片。
    :return:
    """
    # 轮廓检测
    contours, hierarchy = cv.findContours(img_for_contours, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # 绘制轮廓
    image_copy = origin_image.copy()
    cv.drawContours(image_copy, contours, -1, (0, 255, 0), 2)
    return contours  # 返回被标记轮廓的图片复制品


def find_plate_contour(contours: list, original_image):
    """ 找到车牌轮廓

    这个函数用于从轮廓检测中找到车牌所在位置的轮廓。

    :param contours: 一个列表，包含了所有轮廓信息
    :param original_image: 原始图片，用于获取图片副本并在其上面绘制车牌区域轮廓

    :return contour: 一个列表，包含了车牌区域的轮廓信息
    :return image_copy: 一个cv2图片，在上面绘制了车牌区域轮廓

    """
    image_copy = None  # 准备保存返回值。
    contour = None  # 准备保存返回值

    # 筛选车牌位置轮廓
    for index, item in enumerate(contours):
        rect = cv.boundingRect(item)  #
        x = rect[0]
        y = rect[1]
        width = rect[2]
        height = rect[3]
        if (width > height * 2.5) and (width < height * 4):
            # 符合条件的 contour, 保存进contour准备返回
            contour = np.array(item)  # contours中的每个轮廓都是用numpy.array的数据类型保存的
            # print(index)
            image_copy = original_image.copy()  # 获取原图副本
            cv.drawContours(image_copy, contours, 1, (0, 255, 0), 2)  # 在原图绘制车牌所在的边界

    return contour, image_copy  # 返回被绘制轮廓的图片


def fit_straight_line(contour, original_image):
    """ 拟合直线

    该函数将从image_with_contours中的车牌轮廓处获取轮廓点集，然后通过点集拟合直线。
    并且将直线的信息（vx,vy,x,y）以及被标记拟合直线的原图作为结果返回
    含有拟合直线信息的列表用于后续旋转图片，被绘制直线的图片用于阶段性测试

    :param contour: 包含所有轮廓信息的列表
    :param original_image: 原图，用于获取复制以在原图复制品上面标记拟合直线

    :return: 包含拟合直线信息的列表 [vx,vy,x,y]
    :return: 被绘制拟合直线的图片 image_copy

    """
    image_copy = original_image.copy()
    height, width = image_copy.shape  # 获取宽高
    [vx, vy, x, y] = cv.fitLine(contour, cv.DIST_L2, 0, 0.01, 0.01)  # 拟合直线并获取直线信息
    print(f'拟合直线的结果是: {[vx, vy, x, y]}')

    k = vy[0] / vx[0]  # 拟合直线的斜率k
    b = y[0] - k * x[0]  # 拟合直线的b

    print(f'拟合直线的斜率和截距是：k = {k}, b = {b}')  # 拟合直线的表达式

    # 画出拟合直线
    image_copy = cv.line(image_copy, (0, int(b)), (width, int(k * width + b)), (0, 255, 0), 2)
    # 返回结果
    return [vx, vy, x, y], image_copy
