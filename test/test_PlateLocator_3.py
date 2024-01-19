from source import PlateLocator
from source import ToolKit as toolKit

import cv2 as cv

original_image = cv.imread('../resources/test_plate.png')  # 读取初始图片
pl = PlateLocator.PlateLocator(original_image)  # 创建实例

# 输出原图
toolKit.show_color('original image saved in instance', pl.original_image)

# 下面开始手动预处理
'''
pl.gauss_denoise()
pl.grayscale_process()
pl.edge_detect()
toolKit.show_gray('edge_detected image', pl.abs_x)  # show image after edge detected.
pl.adaptive_threshold()
pl.closed_operation()
toolKit.show_gray('image after closed operation', pl.closed_operated_image)
pl.median_filter()
toolKit.show_gray('image after median threshold', pl.median_image)
pl.detect_contours(pl.median_image)
toolKit.show_color('image with all main contours', pl.image_with_contours)
pl.find_plate_contour(pl.contours)
toolKit.show_color('image with contour on plate', pl.image_with_contour)
'''
pl.pre_process()
toolKit.show_color('image after pre process', pl.image_with_contour)

# 下面是图片旋转的部分
'''
pl.fit_straight_line(pl.contour)
toolKit.show_color('image with straight line', pl.image_with_line)
pl.rotate_image(pl.line_info)
toolKit.show_color('image after rotated', pl.rotated_image)
'''

pl.rotate_by_line()
toolKit.show_color('image after rotated by straight line', pl.rotated_image)


# 下面是对新图片预处理的过程
'''
pl.gauss_denoise(pl.rotated_image)
pl.grayscale_process()
pl.edge_detect()
toolKit.show_gray('new edge_detected image', pl.abs_x)  # show image after edge detected.
pl.adaptive_threshold()
pl.closed_operation()
toolKit.show_gray('new image after closed operation', pl.closed_operated_image)
pl.median_filter()
toolKit.show_gray('new image after median threshold', pl.median_image)
pl.detect_contours(pl.median_image, pl.rotated_image)
toolKit.show_color('new image with all main contours', pl.image_with_contours)
pl.find_plate_contour(pl.contours, pl.rotated_image)
toolKit.show_color('new image with contour on plate', pl.image_with_contour)
'''
pl.pre_process(pl.rotated_image)
toolKit.show_color('new image after second pre process', pl.image_with_contour)