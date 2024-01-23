"""
this model is used to locate the plates in the raw image.
after processing, it will return an image witch contains the area of the plate.
这个模块用于在原始图像中定位车牌，在车牌定位处理后，将会返回一个只含有车牌区域的图片。
===== ===== ===== =====
the usage of this class is recommended as follows:
该类的建议使用方式如下：
1. 创建实例

2. 调用快速函数

3. 依次调用内部单一函数


"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
import warnings

from source.ToolKit import *


class PlateLocator:
    """
    This class is used to locate the plates in the original image.
    """
    original_image = None  # 用于保存原始图片

    gauss_image = None  # 高斯处理后的图片备份
    gray_image = None  # 灰度处理后的图片备份
    abs_x = None  # Sobel边缘检测结果的图片备份
    ret = None  # 自适应阈值处理结果备份
    adaptive_image = None  # 自适应阈值处理结果图片备份
    closed_operated_image = None  # 闭运算结果图片备份
    median_image = None  # 中值滤波结果图片备份
    contours = None  # 轮廓检测结果集数据备份
    image_with_contours = None  # 绘制所有轮廓后的图片备份
    contour = None  # 车牌区域结果数据备份
    image_with_contour = None  # 标记了车牌区域的图片备份
    line_info = None  # 拟合直线数据备份，该变量保存一个列表：[vx, vy, x, y]
    image_with_line = None  # 绘制拟合直线后的图片备份
    rotated_image = None  # 旋转图片备份
    plate_image = None  # 车牌区域图片备份

    def __init__(self, original_image):
        """
        构造函数，定义实例的同时传入原始图片。

        :param original_image: 原始图片，推荐使用cv2.imread()获取
        :return: None
        """
        self.original_image = original_image

    def load_original_image(self, original_image):
        """
        构造函数，定义实例的同时传入原始图片。

        :param original_image: 原始图片，推荐使用cv2.imread()获取
        :return: None
        """
        self.original_image = original_image

    def locate_plate(self):
        """
        该函数用于调用类内函数并且一条龙式实现车牌定位。

        :param original_image: 原始图片，建议为 cv2.imread() 读取的 bgr 格式图片
        :return plate_image: 车牌区域图片，用于预览车牌区域，该返回值同时也会被保存到指定目录中。
        """
        pass

    def get_plate_image(self):
        """
        获取车牌区域图片。
        - 调用该函数建议首先调用预处理函数 pre_process() 以及车牌旋转的函数 rotate_by_line()

        :return: 返回车牌区域图片，并且将其保存到指定目录
        """
        self.pre_process(self.rotated_image)  # 对旋转后的图片进行预处理
        # 现在实例中保存的图片都是对旋转后的图片的预处理的结果
        # 接下来就可以进行车牌识别
        rect = cv.boundingRect(self.contour)  # 用最小矩形把车牌区域围起来
        x = rect[0]
        y = rect[1]  # 矩形左上角坐标
        width = rect[2]
        height = rect[3]  # 矩形宽高
        self.plate_image = self.rotated_image[y:y + height, x:x + width]
        cv.imwrite('../attachments/plate_image.png', self.plate_image)

    def rotate_by_line(self, original_image=None):
        """
        通过（拟合）直线旋转图片。
        本函数会通过车牌位置的轮廓，拟合一条直线，并通过该拟合直线信息旋转图片。
        该函数建议在预处理后调用。

        :return: None。被旋转的图片会被保存到指定位置
        """
        # 先进行一次判断
        if original_image is None:
            original_image = self.image_with_contour
        if original_image is None:
            print('拟合直线旋转图片 失败，没有源图片')
            return

        # 拟合直线
        self.fit_straight_line(self.contour, original_image)
        # 旋转图片
        self.rotate_image(self.line_info, original_image)

    def pre_process(self, original_image=None):
        """
        预处理函数，该函数将集成所有处理倾斜车牌使用的函数。
        预处理函数需要在处理图片的第一步使用，目的是将倾斜的车牌旋转为可处理的水平的车牌。
        预处理函数结束后，会在图片中标记出车牌轮廓

        :return: 将旋转结束的车牌赋值给self.rotated_image。
        """
        if original_image is None:
            original_image = self.original_image
        if original_image is None:
            print('出错：预处理 没有有效图片源')
            return

        # 高斯去噪
        self.gauss_denoise(original_image)
        # print('高斯去噪处理完毕')
        # 灰度处理
        self.grayscale_process()  # 灰度检测检测的是高斯去噪的结果图
        # 边缘检测
        self.edge_detect()  # 边缘检测检测的是灰度图
        # 阈值处理
        self.adaptive_threshold(self.abs_x)  # 阈值处理检测的是abs_x
        # 闭运算、去除白点
        self.closed_operation()  # 闭运算检测的是阈值处理的结果图
        # 中值滤波
        self.median_filter()  # 中值滤波处理的是闭运算的结果图
        # 轮廓检测
        self.detect_contours(self.median_image, original_image)
        # 筛选车牌位置
        self.find_plate_contour(self.contours, original_image)

    def gauss_denoise(self, original_image=None):
        """ 高斯去噪处理
        本函数用于对图片进行高斯去噪

        :return: 返回高斯去噪后的结果
        """
        if original_image is None:  # 如果参数缺省，则调用高斯去噪的处理结果。
            original_image = self.original_image
        if original_image is None:
            print('出错：灰度处理没有有效图片源')
            return

        # 高斯去噪：输入原始图像，输出赋值给 self.gauss_image
        self.gauss_image = cv.GaussianBlur(original_image, (3, 3), 0)  # 调用cv2进行高斯去噪处理

    def grayscale_process(self, original_image=None):
        """灰度处理
        本函数用于对原始图像进行灰度处理

        :param original_image: 接受灰度处理的图片，如果不传入图片，默认用类内的高斯处理结果
        :return 如果没有有效图源，打印出错信息，并且结束函数
        :return 操作成功后，将结果赋值给self.gray_image
        """
        if original_image is None:  # 如果参数缺省，则调用高斯去噪的处理结果。
            original_image = self.gauss_image
        if original_image is None:
            print('出错：灰度处理没有有效图片源')
            return

        self.gray_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)  # 进行灰度处理

    def edge_detect(self, original_image=None):
        """ 边缘检测
        对原图像进行边缘检测，返回检测结果。

        :param original_image: 被检测图片
        :return: 返回一个图像，该图像为对原图进行边缘检测的结果。
        """
        if original_image is None:  # 如果参数缺省，则用灰度处理结果。
            original_image = self.gray_image
        if original_image is None:
            print('出错：边缘检测 没有有效图片源')
            return

        sobel_x = cv.Sobel(original_image, cv.CV_16S, 1, 0, ksize=3)  # 进行x方向边缘检测，卷积核尺寸为3
        self.abs_x = cv.convertScaleAbs(sobel_x)  # 将边缘检测结果转换为 unit8 类型
        # return abs_x  # 返回检测结果

    def adaptive_threshold(self, original_image=None):
        """ 自适应阈值处理
        自适应阈值处理。

        :return ret: 返回阈值处理结果
        :return adaptive_image: 阈值处理结果图
        """
        if original_image is None:  # 如果参数缺省，就用边缘检测的结果 abs_x。
            original_image = self.abs_x
        if original_image is None:
            print('出错：自适应阈值处理 没有有效图片源')
            return

        ret, adaptive_image = cv.threshold(original_image, 0, 255, cv.THRESH_OTSU)
        self.ret = ret
        self.adaptive_image = adaptive_image
        # return ret, adaptive_image

    def test_close_operation(self, original_image=None):
        """
        闭运算测试函数，这里没有进行白点去除操作

        :param original_image:
        :return:
        """
        warnings.warn('这是用于测试闭运算的函数，除非进行测试，否则不要使用', DeprecationWarning)
        if original_image is None:  # 如果参数缺省，就用 自适应阈值处理 的结果 self.adaptive_image。
            original_image = self.adaptive_image
        if original_image is None:
            print('出错：闭运算 没有有效图片源')
            return

        # 进行闭运算
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (14, 5))  # 创建卷积核
        closed_operated_image = cv.morphologyEx(original_image, cv.MORPH_CLOSE, kernel, iterations=1)  # 进行闭运算

        return closed_operated_image  # 返回闭运算处理结果

    def closed_operation(self, original_image=None):
        """闭运算
        对原图片进行闭运算，并且去除闭运算后图像中存在的白点

        :param original_image: 原图
        :return: 返回闭运算后的图片
        """
        if original_image is None:  # 如果参数缺省，就用 自适应阈值处理 的结果 self.adaptive_image。
            original_image = self.adaptive_image
        if original_image is None:
            print('出错：闭运算 没有有效图片源')
            return

        # 进行闭运算
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (14, 5))  # 创建卷积核
        closed_operated_image = cv.morphologyEx(original_image, cv.MORPH_CLOSE, kernel, iterations=1)  # 进行闭运算

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

        self.closed_operated_image = image  # 返回处理后的图片

    def median_filter(self, original_image=None):
        """中值滤波
        对原图进行中值滤波

        :param original_image: 原图
        :return: 滤波结果
        """
        if original_image is None:  # 如果参数缺省，闭运算 的结果 self.closed_operated_image。
            original_image = self.closed_operated_image
        if original_image is None:
            print('出错：中值滤波 没有有效图片源')
            return

        self.median_image = cv.medianBlur(original_image, 15)
        # return median_image

    def detect_contours(self, img_for_contours, original_image=None):
        """轮廓检测
        该函数会通过img_for_contours检测出所有轮廓，然后在origin_image中标出轮廓

        :param original_image: 原图片，这个图片用于被标记轮廓，建议使用被处理的图片。
        :param img_for_contours: 应传入中值滤波后的图片
        :return contours: 包含所有轮廓信息的列表。
        :return image_copy: 被标记轮廓的图片（用于检查轮廓检测结果）
        """
        if original_image is None:  # 如果参数缺省，中值滤波 的结果 self.median_image。
            original_image = self.original_image
        if original_image is None:
            print('出错：轮廓检测 没有有效图片源')
            return

        # 轮廓检测
        contours, hierarchy = cv.findContours(img_for_contours, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # 绘制轮廓
        image_copy = original_image.copy()  # 源图片的副本（这里不用self.original_image，是因为该函数后续也会处理被旋转后的图片）
        cv.drawContours(image_copy, contours, -1, (0, 255, 0), 4)

        self.contours = contours
        self.image_with_contours = image_copy  # 保存备份
        # return contours, image_copy  # 返回被标记轮廓的图片复制品

    def detect_contours_copy(self, img_for_contours):
        """函数已废弃：轮廓检测
        该函数会通过img_for_contours检测出所有轮廓，然后在origin_image中标出轮廓

        :param img_for_contours: 应传入中值滤波后的图片
        :param origin_image: 原图片，这个图片用于被标记轮廓，建议使用被处理的图片。
        :return:
        """
        warnings.warn("该函数已废弃（原来用于测试的函数），请使用`detect_contours()`", DeprecationWarning)
        # 轮廓检测
        contours, hierarchy = cv.findContours(img_for_contours, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # 绘制轮廓
        image_copy = self.original_image.copy()
        cv.drawContours(image_copy, contours, -1, (0, 255, 0), 5)
        return contours  # 返回被标记轮廓的图片复制品

    def find_plate_contour(self, contours, original_image=None):
        """ 找到车牌轮廓
        这个函数用于从轮廓检测中找到车牌所在位置的轮廓。

        :param original_image:
        :param contours: 一个列表，包含了所有轮廓信息
        :return contour: 一个列表，包含了车牌区域的轮廓信息
        :return image_copy: 一个cv2图片，在上面绘制了车牌区域轮廓
        """
        if original_image is None:  # 如果参数缺省，原图 的结果 self.original_image。
            original_image = self.original_image
        if original_image is None:
            print('出错：车牌区域轮廓识别 没有有效图片源')
            return

        image_copy = None  # 准备保存返回值。
        contour = None  # 准备保存返回值

        # 筛选车牌位置轮廓
        for index, item in enumerate(contours):
            rect = cv.boundingRect(item)  #
            x = rect[0]
            y = rect[1]
            width = rect[2]
            height = rect[3]
            print('finding...')
            if (width > height * 2.5) and (width < height * 5):
                print('find one....')
                # 符合条件的 contour, 保存进contour准备返回
                contour = np.array(item)  # contours中的每个轮廓都是用numpy.array的数据类型保存的
                # print(index)
                image_copy = original_image.copy()  # 获取原图副本
                cv.drawContours(image_copy, contours, 1, (0, 255, 0), 2)  # 在原图绘制车牌所在的边界

        self.contour = contour
        self.image_with_contour = image_copy
        # return contour, image_copy  # 返回被绘制轮廓的图片

    def fit_straight_line(self, contour, original_image=None):
        """ 拟合直线

        该函数将从image_with_contours中的车牌轮廓处获取轮廓点集，然后通过点集拟合直线。
        并且将直线的信息（vx,vy,x,y）以及被标记拟合直线的原图作为结果返回
        含有拟合直线信息的列表用于后续旋转图片，被绘制直线的图片用于阶段性测试

        :param contour: 包含所有轮廓信息的列表
        :param original_image: 原图，用于获取复制以在原图复制品上面标记拟合直线
        :return: 包含拟合直线信息的列表 [vx,vy,x,y]
        :return: 被绘制拟合直线的图片 image_copy
        """
        if original_image is None:  # 如果参数缺省，原图 的结果 self.original_image。
            original_image = self.original_image
        if original_image is None:
            """
            出现下面的报错是因为没有合适的图片用于绘制拟合直线
            """
            print('出错：拟合直线 没有有效图 绘制 片源')
            return

        image_copy = original_image.copy()
        height, width = image_copy.shape[:2]  # 获取宽高
        [vx, vy, x, y] = cv.fitLine(contour, cv.DIST_L2, 0, 0.01, 0.01)  # 拟合直线并获取直线信息
        print(f'拟合直线的结果是: {[vx, vy, x, y]}')

        k = vy[0] / vx[0]  # 拟合直线的斜率k
        b = y[0] - k * x[0]  # 拟合直线的b

        print(f'拟合直线的斜率和截距是：k = {k}, b = {b}')  # 拟合直线的表达式

        # 画出拟合直线
        image_copy = cv.line(image_copy, (0, int(b)), (width, int(k * width + b)), (0, 255, 0), 2)
        # 返回结果
        self.line_info = [vx, vy, x, y]
        self.image_with_line = image_copy
        # return [vx, vy, x, y], image_copy

    def rotate_image(self, fit_line_info, original_image=None):
        """ 旋转图片。

        这个函数通过从 fit_line_info 中计算图片（中的车牌区域）的倾斜角度，
        将原始图片（的副本）进行旋转，以保证得到一个具有水平车牌区域的原始图片的副本图片。

        :param fit_line_info: 拟合直线的信息，该参数可以通过 fit_straight_line() 获取
        :return image_copy: 原始图片的副本，是一个经过旋转后实现的车牌水平的图片，该图片的长宽均为原图的1.1倍
        """
        if original_image is None:  # 如果参数缺省，原图 的结果 self.original_image。
            original_image = self.original_image
        if original_image is None:
            print('出错：旋转图片 没有有效图片源')
            return

        [vx, vy, x, y] = fit_line_info  # 从参数中解包拟合直线信息
        k = vy[0] / vx[0]  # 拟合直线的斜率k
        b = y[0] - k * x[0]  # 拟合直线的截距b
        a = math.atan(k)  # 获取倾斜角度（弧度制）
        a = math.degrees(a)  # 获取倾斜角度（角度制）
        image_copy = self.original_image.copy()  # 获取原图像副本
        height, width = self.original_image.shape[:2]  # 获取宽高
        rotation_matrix = cv.getRotationMatrix2D((width / 2, height / 2), a, 0.8)  # 获取仿射矩阵
        image_copy = cv.warpAffine(image_copy, rotation_matrix, (int(width * 1.1), int(height * 1.1)))  # 进行仿射变换

        self.rotated_image = image_copy
        # return image_copy  # 返回旋转后的图片
