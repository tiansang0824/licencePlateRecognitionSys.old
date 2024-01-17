"""
This file contains some useful tools for the project,
the detailed list will be shown below.
=====
这个文件中包含一些为了项目运行而准备的函数，具体的函数列表将会在下方列举出来。

"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Simsun'  # 设定plt显示图片的标题字体，避免中文乱码


def cv_show(win_title, img):
    """
    本函数用于调用cv2的图片显示函数，实现图片输出
    :param win_title: 输出窗口的窗口名
    :param img: 被输出图片
    :return: None
    """
    cv.imshow(win_title, img)  # 调用cv2的图片显示函数
    cv.waitKey(0)  # 用于保证窗口存在
    cv.destroyAllWindows()  # 用户相应后关闭所有显示窗口


def show_color(name, img):
    """
    调用plt显示彩色图像（三通道图像）
    :param name: 显示图片名
    :param img: 被输出的图片
    :return: None
    """
    plt.imshow(img[:, :, ::-1])  # 以彩色图像的形式输出图片
    plt.title(name)  # 设置图片标题
    plt.show()  # 显示图片


def show_gray(name, img):
    """
    调用plt显示灰度图
    :param name: 显示名称
    :param img: 输出图像
    :return: None
    """
    plt.imshow(img, cmap='gray')  # 设置输出图片和输出模式（灰度图）
    plt.title(name)  # 设定图片标题
    plt.show()  # 显示图片
