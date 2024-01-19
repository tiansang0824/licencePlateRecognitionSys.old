# README

This project is used for graduation design, and is for license plate recognition.

此项目用于我的毕设，主要用于车牌识别（电动车车牌识别系统）。

## 项目目录

本项目主要目录如下：

```text
project/
	|- attachments/
	|- resources/
	|- source/
	|- templates/
	|- test/
	|- main.py
	|- README.md
```

- atatchments：主要用于存放程序产生的数据；
- resources：主要用于存放程序需要的其他资源；
- source：用于存放除了程序入口外的其他程序；
- templates：用于存放模板匹配时用到的模板；
- test：用于存放测试文件（主要是jupyter文件）；
- main.py：程序启动入口；
- README.md：说明性文件，即本文件。

## 模块说明

### PlateLocator

该模块主要用于定位车牌，使用时需要首先创建一个实例：

```python
original_image = cv2.imread('original_path')
plate_locator = PlateLocator(original_image)
```

然后调用函数：

```python
plate_locator.locate_plate()
```

即可将车牌部分图片保存到指定目录。

---

PlateLocator的实现流程主要如下：

```mermaid
graph TD
	pre(创建实例并加载图片)-->a
	a(高斯去噪) --> b(灰度处理)
	b-->c(sobel算子边缘检测)
	c-->d(自适应阈值处理)
	d-->e(闭运算，让白色区域连为一体)
	e-->f(去除白点)
	f-->g(中值滤波去除噪点)
	g-->h(轮廓检测)
	h-->i(筛选车牌位置轮廓)
	i-->j(拟合直线)
	j-->k(旋转图片)
	k-->a
	i-->l(（对于旋转后的图片）获取车牌区域坐标和长宽)
	l-->m(返回和保存车牌区域图片)
```

