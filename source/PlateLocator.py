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


def find_plate_contour(contours, original_image):
    """
    该图片用于筛选车牌位置的轮廓，并且将其绘制到原图中，最后返回被绘制车牌轮廓的部分
    :param contours: 包含轮廓信息的列表
    :param original_image: 被绘制轮廓的列表
    :return: 返回被绘制轮廓后的图片
    """
    image_copy = None  # 准备保存返回值。
    # 筛选车牌位置轮廓
    for index, item in enumerate(contours):
        rect = cv.boundingRect(item)
        x = rect[0]
        y = rect[1]
        width = rect[2]
        height = rect[3]
        if (width > height * 2.5) and (width < height * 4):
            # print(index)
            image_copy = original_image.copy()
            cv.drawContours(image_copy, contours, 1, (0, 255, 0), 2)
    return image_copy  # 返回被绘制轮廓的图片

# TODO: 拟合直线找斜率函数。
