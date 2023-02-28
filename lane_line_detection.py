# -*- coding: utf-8 -*- 

import cv2
import numpy as np
from pytest import param


class Laneline:

    def __init__(self, 
                gaussian_ksize,    # 高斯模糊卷积核
                gaussian_sigmax,   # 高斯模糊x方向上的高斯核标准偏差
                canny_threshold1,  # 边缘检测第一个阈值
                canny_threshold2,  # 边缘检测第二个阈值，用于检测图像中明显的边缘
                slope_diff,        # 线段斜率差值，用来剔除斜率不一致的线段
                line_threshold,    # 直线阈值，表示确定一条直线至少需要多少个曲线相交
                linelength_min,    # 最小直线长度，小于该长度就抛弃
                linegap_max        # 最大直线断裂值，超过了设定值，则把两条线段当成一条线段，值越大，允许线段上的断裂越大，越有可能检出潜在的直线段
                ):      
        
        self.gaussian_ksize = gaussian_ksize
        self.gaussian_sigmax = gaussian_sigmax
        self.canny_threshold1 = canny_threshold1
        self.canny_threshold2 = canny_threshold2
        self.slope_diff = slope_diff
        self.line_threshold = line_threshold
        self.linelength_min = linelength_min
        self.linegap_max = linegap_max

    def _image_process(self, color_img):
        """
        @function: 高斯模糊,灰度化,canny边缘检测
        @param:    原始帧bgr图像
        @return:   edges
        """
        img = cv2.GaussianBlur(color_img, (self.gaussian_ksize, self.gaussian_ksize), self.gaussian_sigmax)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.canny_threshold1, self.canny_threshold2)
        return edges


    def _get_roi_mask(self, gray_img):
        """
        @function: 获取车道线的感兴趣区域
        @param:    gray_img
        @return:   mask
        """
        vertices = np.array([[[0, 368], [300, 210], [340, 210], [640, 368]]]) # 掩码切片
        mask = np.zeros_like(gray_img)
        mask = cv2.fillPoly(mask, vertices, color = 255)
        roi_mask = cv2.bitwise_and(gray_img, mask) # 布尔与运算
        return roi_mask


    def _get_lines(self, roi_image):
        """
        @function: 获取感兴趣区域中的所有线段
        @param:    roi_image,标记边缘的灰度图
        @return:   lines
        """

        def calculate_slope(line):
            """
            @function: 计算线段line的斜率
            @param:    line, np.array([[x_1, y_1, x_2, y_2]])
            @return:   k
            """
            x1, y1, x2, y2 = line[0]
            if x1 != x2:
                k = (y2 - y1) / (x2 - x1)
            else:
                k = 1000000
            return k

        def reject_abnormal_lines(lines):
            """
            @function: 剔除斜率不一致的线段
            @param:    lines: 线段集合, [np.array([[x_1, y_1, x_2, y_2]]),np.array([[x_1, y_1, x_2, y_2]]),...,np.array([[x_1, y_1, x_2, y_2]])]
            @return:
            """
            slopes = [calculate_slope(line) for line in lines]
            while len(lines) > 0:
                mean = np.mean(slopes)
                diff = [abs(s - mean) for s in slopes]
                idx = np.argmax(diff)
                if diff[idx] > self.slope_diff:
                    slopes.pop(idx)
                    lines.pop(idx)
                else:
                    break
            return lines

        def least_squares_fit(lines):
            """
            @function: 将lines中的线段拟合成一条线段
            @param:    lines: 线段集合
            @return:   线段上的两点, np.array([[xmin, ymin], [xmax, ymax]])
            """
            x_coords = np.ravel([[line[0][0], line[0][2]] for line in lines])
            y_coords = np.ravel([[line[0][1], line[0][3]] for line in lines])
            poly = np.polyfit(x_coords, y_coords, deg = 1)
            point_min = (np.min(x_coords), np.polyval(poly, np.min(x_coords)))
            point_max = (np.max(x_coords), np.polyval(poly, np.max(x_coords)))
            points = np.array([point_min, point_max], dtype = np.int)
            return points

        #* 获取所有线段
        lines = cv2.HoughLinesP(roi_image, 1, np.pi / 180, self.line_threshold, minLineLength = self.linelength_min, maxLineGap = self.linegap_max)
        if lines is None:
            print("Not enough lines found in the image")
            exit(0)

        # 按照斜率分成左右车道线
        left_lines = [line for line in lines if calculate_slope(line) > 0]
        right_lines = [line for line in lines if calculate_slope(line) < 0]
        # 剔除离群线段
        left_lines = reject_abnormal_lines(left_lines)
        right_lines = reject_abnormal_lines(right_lines)
        # 聚合车道线
        left_lane_lines = least_squares_fit(left_lines)
        right_lane_lines = least_squares_fit(right_lines)

        return left_lane_lines, right_lane_lines

    def _get_vanish_point(self, lines):

        def _calculate_line_param(line):
            x1, y1, x2, y2 = line[0][0], line[0][1], line[1][0], line[1][1]
            # y = kx + c
            if x1 != x2:
                k = (y2 - y1) / (x2 - x1)
            else:
                k = 1000000
            c = y2 - k * x2
            return k, c
        
        line_param = []
        vanish_point = None
        left_line, right_line = lines
        k1, c1 = _calculate_line_param(left_line)
        k2, c2 = _calculate_line_param(right_line)
        param = [k1,c1,k2,c2]
        line_param.append(param)
        if k1 != k2:
            x0 = (c1 - c2) / (k2 - k1)
            y0 = k1 * x0 + c1
            vanish_point = [x0, y0]

        return vanish_point, line_param


    def _draw_results(self, img, lines, vanish_point, line_param):
        """
        @function：    在img上绘制lines
        @param：       img
        @param：       lines: 两条线段: [np.array([[xmin1, ymin1], [xmax1, ymax1]]), np.array([[xmin2, ymin2], [xmax2, ymax2]])]
        :return:
        """
        def _calculate_euclidean_distance(c,c0):# 计算欧式距离
            return pow((c[0]-c0[0])**2+(c[1]-c0[1])**2, 0.5)
        
        left_line, right_line = lines                                                           # 左右车道线
        lline_point0 = left_line[0]                                                             # 左车道线上点
        lline_point1 = left_line[1]                                                             # 左车道线下点
        rline_point0 = right_line[1]                                                            # 右车道线上点
        rline_point1 = right_line[0]                                                            # 右车道线下点
        lline_k = line_param[0][0]
        lline_b = line_param[0][1]
        rline_k = line_param[0][2]
        rline_b = line_param[0][3]
        
        lline_x0, rline_x0, lline_y0 = 290, 352, 230
        cv2.circle(img, (int(vanish_point[0]), int(vanish_point[1])), 6, (0, 0, 255), -1)

        cv2.circle(img,(lline_x0, lline_y0), 6, (255, 0, 0), -1)
        cv2.circle(img,(rline_x0, lline_y0), 6, (255, 0, 0), -1)
        cv2.circle(img,(int(lline_point1[0]), int(lline_point1[1])), 6, (255, 0, 0), -1)
        cv2.circle(img,(int(lline_point1[0]-480), int(lline_point1[1])), 6, (255, 0, 0), -1)
        
        roi = [(lline_x0, lline_y0), (rline_x0, lline_y0), (lline_point1), (lline_point1[0]-480, lline_point1[1])]
        cv2.fillPoly(img, [np.array(roi)], (0,255,0))

        cv2.line(img, tuple(lline_point0), tuple(lline_point1), (0, 255, 255), 2)
        cv2.line(img, tuple(rline_point1), tuple(rline_point0), (0, 255, 255), 2)

    def _lane_line_detection(self, color_img):
        #* 图像预处理
        cv2.imshow("orig", color_img)
        edges = self._image_process(color_img)  
        #cv2.imshow("edges", edges)
        #* 获取感兴趣区域
        roi_mask = self._get_roi_mask(edges)    
        #cv2.imshow("roi_mask", roi_mask)
        #* 提取车道线
        lines = self._get_lines(roi_mask)    
        #print(lines)
        #* 获取消失点
        vanish_point, line_param = self._get_vanish_point(lines)
        #* 绘制结果
        self._draw_results(color_img, lines , vanish_point, line_param)     
        return vanish_point



if __name__ == '__main__':

    capture = cv2.VideoCapture("/home/lxj/桌面/lane_line_sfm/video/video.mp4")
    #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #outfile = cv2.VideoWriter('output.avi', fourcc, 25., (1280, 368))
    while capture.isOpened():
        ret, frame = capture.read()
        #origin = np.copy(frame)
        cv2.imshow('orig', frame)
        laneline = Laneline(gaussian_ksize = 5, 
                            gaussian_sigmax = 1, 
                            canny_threshold1 = 50, 
                            canny_threshold2 = 100,
                            slope_diff = 0.2,
                            line_threshold = 15,
                            linelength_min = 60,
                            linegap_max = 20)
        vanish_point = laneline._lane_line_detection(frame)
        #output = np.concatenate((origin, frame), axis=1)
        #outfile.write(output)
        cv2.imshow('video', frame)
        # 处理退出
        speed = 100
        if cv2.waitKey(speed) == ord('q'):
            cv2.destroyAllWindows()
            break
