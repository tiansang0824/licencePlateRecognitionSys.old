"""
车牌定位器Locator_v1，该定位器主要通过车牌宽高比例定位车牌位置。

"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Simsun'  # 修改字体,避免中文乱码.


class Locator:
    img_path = ''

    img_height = 0
    img_width = 0

    img = []
    gray = []
    gaussian = []
    median = []
    sobel = []
    binary = []
    dilation = []
    dilation2 = []
    erosion = []
    erosion2 = []
    closed = []
    result = []

    def __init__(self, img_path):
        self.img = cv.imread(img_path)  # 读取原始图片保存到实例中
        self.img_height = self.img.shape[0]  # 读取长度和宽度
        self.img_width = self.img.shape[1]

    def load_image(self, img_path):
        """
        重新读取图片的函数
        :param img_path:
        :return:
        """
        self.img = cv.imread(img_path)

    def pre(self, img):
        """
        预处理函数,主要用于去除噪声

        预处理步骤：
        1. 灰度化
        2. 高斯去噪

        :return:
        """
        # 1. 高斯去噪
        self.gaussian = cv.GaussianBlur(img, (3, 3), 0)
        # 2. 灰度处理
        self.gray = cv.cvtColor(self.gaussian, cv.COLOR_BGR2GRAY)
        # 3. 边缘检测
        sobelx = cv.Sobel(self.gray, cv.CV_16S, 1, 0, ksize=3)
        sobely = cv.Sobel(self.gray, cv.CV_16S, 0, 1, ksize=3)
        gradient = cv.subtract(sobelx, sobely)
        self.sobel = cv.convertScaleAbs(gradient)
        cv.imshow('sobel', self.sobel)
        cv.waitKey(0)
        # 4. 阈值处理
        # 计算最小灰度值
        min_val = np.amin(self.sobel)
        # 计算最大灰度值
        max_val = np.amax(self.sobel)
        # 计算中间值
        mid = (max_val - min_val) / 2 + min_val
        [ret, self.binary] = cv.threshold(self.sobel, mid, 255, cv.THRESH_BINARY)
        cv.imshow('ret', self.binary)
        cv.waitKey(0)
        # 5. 闭运算,去除白点
        # ksize测试数据 (14,5)
        # -> (8,8)会出现很多小空洞, 导致后续腐蚀和膨胀的时候出现大空洞
        # --> (10, 10)空洞几乎不存在,但是会导致检测到的区域多出一小块
        # --> (9,9)
        # 总体上(10,10)的效果比(9,9)好一点
        #
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (14, 10))
        self.closed = cv.morphologyEx(self.binary, cv.MORPH_CLOSE, kernel)
        cv.imshow('closed', self.closed)
        cv.waitKey(0)
        # 5.2. 去除白点
        # 创建卷积核
        kernelx = cv.getStructuringElement(cv.MORPH_RECT, (20, 1))
        kernely = cv.getStructuringElement(cv.MORPH_RECT, (1, 19))
        # 膨胀,腐蚀
        # 先腐蚀在膨胀
        self.erosion = cv.erode(self.closed, kernelx)
        self.dilation = cv.dilate(self.erosion, (25, 1))
        cv.imshow('erosion', self.dilation)
        cv.waitKey(0)
        # 腐蚀,膨胀
        self.erosion2 = cv.erode(self.erosion, kernely)
        self.dilation2 = cv.dilate(self.erosion2, (25, 1))
        cv.imshow('dilation 2', self.dilation2)
        cv.waitKey(0)
        # 6. 中值滤波
        self.median = cv.medianBlur(self.dilation2, 15)
        cv.imshow('median', self.median)
        cv.waitKey(0)

        """
        第一版定位:通过轮廓检测进行定位,功能尚不完善
                # 7. 轮廓检测
        [contours, hierarchy] = cv.findContours(self.median, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # 获取到了所有轮廓信息
        # 绘制轮廓
        img_copy = self.img.copy()
        cv.drawContours(img_copy, contours, -1, (0, 255, 0), 3)
        cv.imshow('img with contours', img_copy)
        cv.waitKey(0)
        # 8. 筛选车牌位置
        image_copy = self.img.copy()
        for index, item in enumerate(contours):
            rect = cv.boundingRect(item)  #
            x = rect[0]
            y = rect[1]
            width = rect[2]
            height = rect[3]
            print('finding...')
            if (width > height * 2.5) and (width < height * 4.5):
                print('find one....')
                # 符合条件的 contour, 保存进contour准备返回
                contour = np.array(item)  # contours中的每个轮廓都是用numpy.array的数据类型保存的
                # print(index)
                cv.drawContours(image_copy, contours, 1, (0, 255, 0), 2)  # 在原图绘制车牌所在的边界
        cv.imshow('rectangle', image_copy)
        cv.waitKey(0)
        """
        """
        第二版定位: 通过直方图低谷确定车牌区域
        """

        # 对图像进行黑白翻转
        # img_reverse 表示翻转结果,
        # 由于是暂时的函数测试,所以 img_reverse 还没有加入到类的成员变量中.
        img_reverse = cv.bitwise_not(self.median)
        cv.imshow('inverse', img_reverse)
        cv.waitKey(0)
        # 准备绘制直方图的数据
        rows = self.img_height
        cols = self.img_width
        print(f'rows: {rows}, cols: {cols}')

        # 统计y方向每一行的黑色像素点个数
        # 并且绘制y方向灰度直方图
        histogram_y = []
        for row in range(rows):
            black_count = 0
            for col in range(cols):
                if img_reverse[row][col] == 0:
                    black_count += 1
            histogram_y.append(black_count)
        print(f'len(histogram_y): {len(histogram_y)}')
        y = [y for y in range(rows)]
        x = histogram_y
        plt.barh(y, x, color='black', height=1)  # 绘图
        # 设置x，y轴标签
        plt.xlabel('黑色像素点个数')
        plt.ylabel('行')
        # 设置刻度
        plt.xticks([x for x in range(0, max(histogram_y) + 10, 5)])
        plt.yticks([y for y in range(0, rows, 20)])
        plt.show()

        # 统计和绘制x方向灰度直方图
        # 统计个数
        histogram_x = []
        for col in range(cols):
            black_count = 0
            for row in range(rows):
                if img_reverse[row][col] == 0:
                    black_count += 1
            histogram_x.append(black_count)
        print(f'for histogram in X, len(histogram_x): {len(histogram_x)}')
        # 绘制直方图
        x = [x for x in range(cols)]
        y = histogram_x
        plt.bar(x, y, color='black', width=1)
        # 设置x,y标签
        plt.xlabel('col')
        plt.ylabel('0 nums')
        # 设置刻度
        plt.xticks([x for x in range(0, cols, 25)])
        plt.yticks([y for y in range(0, max(histogram_x) + 10, 5)])
        plt.title('histogram_x')
        plt.show()

        # 先用x方向的柱状图做一次测试,成功的话就包装成函数实现.
        sum_x = 0  # 计算水平方向的黑色像素块个数,为计算平均值和标准差做准备
        for col in range(cols):
            sum_x += histogram_x[col]  # 加和
        avg_x = (sum_x / len(histogram_x)) / 4  # 计算四分之一均值,用于尽可能剔除无效数据
        hist_x = histogram_x  # 复制一个副本
        for col in range(cols):
            if hist_x[col] < avg_x:
                hist_x[col] = 0  # 剔除较小的无效数据

        print('剔除较小的无效数据,结束,我还活着')
        print(f'hist_x = {hist_x}')
        """
        找到每一个波峰的范围
        
        edges 是一个2维列表,第一纬用于保存所有的边缘信息;
        第二维用于保存成对的点集,点集中第一个点是波峰的左侧起点坐标,
        第二个点是右侧终点坐标.
        
        yi因为以及做过小数据剔除,所以波峰左右两侧一定是0
        
        判断左侧索引:
            左侧索引的标志是该点处的值(黑像素点个数)为0,但是该点右侧的值不为零.
            为了避免下标越界,可以每次判断下一个点的值的标志:
            左侧索引的下一个索引的标志:
            该点的值大于左侧的值,但是该点左侧的值为0
            当满足上述条件时,该点左侧的点就是左侧边界点
        
        判断右侧索引:
            该点处的值为0,并且该点的值小于该点左侧的值
            
        """
        edges_x = []
        edge = []  # 每次循环开始的时候edge都是空集
        for col in range(cols):
            # 判断左侧索引: 1. 该点的值为0; 2. 该点右侧的值不为0
            if hist_x[col - 1] == 0 and hist_x[col] != 0:
                edge.append(col - 1)  # 添加左侧索引
            # 判断右侧索引: 1. 该点值为0; 2. 该点左侧值不为0
            if hist_x[col] == 0 and hist_x[col - 1] != 0:
                edge.append(col)  # 添加右侧索引
            if len(edge) == 2:
                print(f'success 1')
                # 集齐了一个波峰的坐标
                edges_x.append(edge)  # 将边缘信息追加到edges列表中.
                edge = []

        print('获取边界集合,结束,我还活着')
        print(f'边界集 {edges_x}')

        """
        经过上述循环后,edges中所有的点集就是每一个波峰的左右分界点
        
        接下来就可以通过计算标准差和变异系数判断每个波峰的平稳程度,
        其中,变化最小的就是车牌所属区域.
        """
        for part_index in edges_x:
            left_index = part_index[0]
            right_index = part_index[1]
            part = histogram_x[left_index:right_index:1]  # 某一个波峰
            std_dev = np.std(part)  # 计算标准差
            print(f"Standard deviation is {std_dev}")  # 打印标准差
        print('打印所有波峰标准差,结束,我还活着')

    def show_process(self):
        cv.imshow('img', self.img)
        cv.imshow('gray', self.gray)
        cv.imshow('gaussian', self.gaussian)
        cv.imshow('median', self.median)
        cv.imshow('sobel', self.sobel)
        cv.imshow('binary', self.binary)
        cv.imshow('dilation', self.dilation)
        cv.imshow('erosion', self.erosion)
        cv.imshow('dilation2', self.dilation2)
        cv.imshow('closed', self.closed)
        cv.waitKey(0)


if __name__ == '__main__':
    locator = Locator('test.png')
    locator.pre(locator.img)
