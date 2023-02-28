# -*- coding: utf-8 -*- 

import cv2
import pangolin
import numpy as np
import math

from map import Map
from lane_line_detection import Laneline
from bundle_adjustment import BundleAdjustment

from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform

class Frame(object):

    idx = 0
    last_kps, last_des, last_pose = None, None, None

    def __init__(self, image):
        """
        只要一经初始化，Frame 就会把上一帧的信息传递给下一帧
        """
        Frame.idx += 1

        self.image = image
        self.idx  = Frame.idx
        self.last_kps  = Frame.last_kps
        self.last_des  = Frame.last_des
        self.last_pose = Frame.last_pose


# 利用相机内参对角点的像素坐标进行归一化
def normalize(K, pts):
    Kinv = np.linalg.inv(K)     # 矩阵求逆
    # turn [[x,y]] -> [[x,y,1]]
    add_ones = lambda x: np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    norm_pts = np.dot(Kinv, add_ones(pts).T).T[:, 0:2]
    return norm_pts

class Slam:

    def __init__(self, width, height, fov, matches_ratio):
        
        self.width = width
        self.height = height
        self.fov = fov
        self.matches_ratio = matches_ratio

        # 地图结构
        self.vanish_points = []                     # 消失点

        # 相机参数
        self.camera_xyz = np.matrix([0, 0, 0]).I
        self.world_scale = 10

        # 相机内参K
        cx = width / 2
        cy = height / 2
        fov = fov * (math.pi / 180)
        fx = cx / math.tan(fov / 2)
        fy = cy / math.tan(fov / 2)
        self.K = np.array( [[fx, 0, cx],
                            [0, fy, cy],
                            [0,  0,  1]])
    
    '''
    description: 特征提取
    param: self
    param: frame
    return: kps 角点在图像中坐标，
            des 角点的描述子，一般为32维的特征向量
    '''    
    def _extract_points(self, frame):
        """
        @function: 提取角点 orb/sift
        @param:    frame
        @return:   kps, des 
        """
        image = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)

        ''' 1.(FAST + BRIEF)
        orb = cv2.ORB_create()
        kps, des = orb.detectAndCompute(image, None)
        '''
        
        ''' 2.(ShiTomas + BRIEF)
        '''
        orb = cv2.ORB_create()
        mask = np.zeros(image.shape[:2], dtype=np.uint8)       
        mask[0:image.shape[0], 0:image.shape[1]] = 255                
        pts = cv2.goodFeaturesToTrack(image, 2000, qualityLevel = 0.01, minDistance = 3, mask=mask)
        kps = [cv2.KeyPoint(x = pt[0][0], y = pt[0][1], _size = 20) for pt in pts]
        kps, des = orb.compute(image, kps)
        
        

        ''' 3. sift
        sift = cv2.SIFT_create()
        kps, des = sift.detectAndCompute(image, None)
        '''

        ''' 4.(FAST + BEBLID)
        orb = cv2.ORB_create()
        descriptor = cv2.xfeatures2d.BEBLID_create(0.75)
        kps = orb.detect(image, None)
        kps, des = descriptor.compute(image, kps)
        '''

        ''' 5.(ShiTomas + BEBLID)
        descriptor = cv2.xfeatures2d.BEBLID_create(0.75)
        pts = cv2.goodFeaturesToTrack(image, 2000, qualityLevel = 0.01, minDistance = 3)
        kps = [cv2.KeyPoint(x = pt[0][0], y = pt[0][1], _size = 20) for pt in pts]
        kps, des = descriptor.compute(image, kps)
        '''
    
        kps = np.array([(kp.pt[0], kp.pt[1]) for kp in kps])
        return kps, des

    '''
    description: 特征匹配
    param {*} self
    param {*} frame
    return {*} match_kps
    '''    
    def _match_points(self, frame):
        """
        @function: 当前帧的角点和上一帧的进行配准 
        @param:    frame
        @return:   match_kps 
        """
        bfmatch = cv2.BFMatcher(cv2.NORM_HAMMING) # orb匹配
        #bfmatch = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING) # BEBLID匹配
        #bfmatch = cv2.BFMatcher() # sift匹配
        
        matches = bfmatch.knnMatch(frame.curr_des, frame.last_des, k = 2)
        match_kps, idx1, idx2 = [], [], []

        for m,n in matches:
            if m.distance < self.matches_ratio * n.distance:
                idx1.append(m.queryIdx)
                idx2.append(m.trainIdx)

                p1 = frame.curr_kps[m.queryIdx]          # 当前帧配准的角点位置
                p2 = frame.last_kps[m.trainIdx]          # 上一帧配置的角点位置
                match_kps.append((p1, p2))
        
        assert len(match_kps) >= 8
        #print(len(match_kps))

        frame.curr_kps = frame.curr_kps[idx1]
        frame.last_kps = frame.last_kps[idx2]

        return match_kps

    # 计算本质矩阵
    def _calculate_essential_matrix(self, match_kps):
        """
        @function: 八点法对本质矩阵求解
        @param:    match_kps
        @return:   essential_matrix 
        """
        match_kps = np.array(match_kps)

        # 使用相机内参对角点像素坐标归一化
        norm_curr_kps = normalize(self.K, match_kps[:, 0])
        norm_last_kps = normalize(self.K, match_kps[:, 1])

        # 求解本质矩阵和内点数据
        #cv2.findEssentialMat
        model, inliers = ransac((norm_last_kps, norm_curr_kps),
                                EssentialMatrixTransform,
                                min_samples = 8, # 最少需要8个点
                                residual_threshold = 0.005,
                                max_trials = 200)

        frame.curr_kps = frame.curr_kps[inliers] # 保留当前帧的内点数据
        frame.last_kps = frame.last_kps[inliers] # 保留上一帧的内点数据

        return model.params # 返回本质矩阵

    # 从本质矩阵中分解出相机运动 R、t
    def _extract_Rt(self, E):
        """
        @function: 从E中分解出相机运动R、t
        @param:    E
        @return:   Rt 
        """
        W = np.mat([[0,-1,0],[1,0,0],[0,0,1]],dtype=float)
        U,d,Vt = np.linalg.svd(E)

        if np.linalg.det(U)  < 0: U  *= -1.0
        if np.linalg.det(Vt) < 0: Vt *= -1.0

        # 相机没有转弯，因此R的对角矩阵非常接近 diag([1,1,1])
        R = (np.dot(np.dot(U, W), Vt))
        if np.sum(R.diagonal()) < 0:
            R = np.dot(np.dot(U, W.T), Vt)

        t = U[:, 2]     # 相机一直向前，分量t[2] > 0
        if t[2] < 0:
            t *= -1

        Rt = np.eye(4)
        Rt[:3, :3] = R
        Rt[:3, 3] = t
        return Rt, R, t      # Rt为从相机坐标系的位姿变换到世界坐标系的位姿

    # opencv的三角测量函数
    def _cv_triangulate(self, pts1, pts2, pose1, pose2):
        pts1 = normalize(self.K, pts1)
        pts2 = normalize(self.K, pts2)

        pose1 = np.linalg.inv(pose1)
        pose2 = np.linalg.inv(pose2)

        points4d = cv2.triangulatePoints(pose1[:3], pose2[:3], pts1.T, pts2.T).T
        points4d /= points4d[:, 3:]
        return points4d
   

    '''
    description: 给定投影矩阵pose1,pose2和图像上的匹配特征点pts1,pts2,从而计算三维点坐标
    param {*} pts1, pts2, pose1, pose2
    return {*} points4d 
    '''
    # 三角测量函数
    def _triangulate(self, pts1, pts2, pose1, pose2):

        #points3d = []
        pose1_inv = np.linalg.inv(pose1)                    # 从世界坐标系变换到相机坐标系的位姿, 因此取逆
        pose2_inv = np.linalg.inv(pose2)
        pts1 = normalize(self.K, pts1)                      # 使用相机内参对角点坐标归一化
        pts2 = normalize(self.K, pts2)

        points4d = np.zeros((pts1.shape[0], 4))             # Nx4的N维数组
        
        for i, (kp1, kp2) in enumerate(zip(pts1, pts2)):
            A = np.zeros((4,4))                             # 构造参数矩阵A
            A[0] = kp1[0] * pose1_inv[2] - pose1_inv[0]
            A[1] = kp1[1] * pose1_inv[2] - pose1_inv[1]
            A[2] = kp2[0] * pose2_inv[2] - pose2_inv[0]
            A[3] = kp2[1] * pose2_inv[2] - pose2_inv[1]
            _, _, vt = np.linalg.svd(A)                     # 对A进行奇异值分解
            points4d[i] = vt[3]

        points4d /= points4d[:, 3:]                         # 归一化变换成齐次坐标 [x, y, z, 1]
        return points4d

    # 画出角点的运动轨迹
    def _draw_points(self, frame):
        """
        @function: 画出角点的运动轨迹
        @param:    frame
        @return:   None 
        """
        for kp1, kp2 in zip(frame.curr_kps, frame.last_kps):
            u1, v1 = int(kp1[0]), int(kp1[1])
            u2, v2 = int(kp2[0]), int(kp2[1])
            cv2.circle(frame.image, (u1, v1), color=(0,0,255), radius=2)
            cv2.line(frame.image, (u1, v1), (u2, v2), color=(255,0,0))
        return None

    '''
    description: 筛选角点
    param {*} self
    param {*} Rt
    param {*} points4d 多维数组Nx4,相邻的两帧三角化出多个地图点
    return {*}
    R: [[ 9.99999998e-01  2.88254823e-05 -5.55796806e-05]
        [-2.87518461e-05  9.99999123e-01  1.32442180e-03]
        [ 5.56178089e-05 -1.32442020e-03  9.99999121e-01]]
    t:  [0.49943141 0.44633816 0.74252981]
    '''    
    def _check_points(self, kp1, kp2, R, t, points4d):
        #print("R:{}\n t:{}\n".format(R, t))
        fx = self.K[0][0]
        cx = self.K[0][2]
        fy = self.K[1][1]
        cy = self.K[1][2]
        point_array = np.hstack((points4d,kp1))
        points3d1 = []                                                  #& 参考帧的三维地图点列表
        points3d2 = []                                                  #& 当前帧的三维地图点列表
        points3d_param = []                                             #& 每帧的三维地图点参数列表
        points3d_param_array = np.zeros((len(points4d), 3))             #& 三维地图点参数N维数组
        R_trans = np.transpose(R)
        t_trans = np.transpose([t])                                     #& 3x1矩阵
        O1 = np.zeros((3,1), np.float32)                                #& 第一个相机的光心设置为世界坐标系下的原点
        O2 = -R_trans * t_trans                                         #& 第二个相机的光心位置在世界坐标系下的坐标
        for point in point_array:                                       #& 去归一化，转为非齐次坐标 [x,y,z,1] to [x,y,z]
            p3dC1 = np.array([point[0], point[1], point[2]])            #& 获得每帧中的每个地图点
            p3dC2 = R*np.transpose([p3dC1])+t_trans
            points3d1.append(p3dC1)  
            points3d2.append(p3dC2)

            # 计算视差角                                    
            normal1 = np.transpose([p3dC1]) - O1                        #& 3x1向量PO1，地图点指向第一个相机的光心
            normal2 = np.transpose([p3dC1]) - O2                        #& 3x1向量PO1，地图点指向第二个相机的光心
            normal1_tran = np.transpose(normal1)
            dist1 = cv2.norm(normal1)                                   #& 地图点指向第一个相机光心的向量距离
            dist2 = cv2.norm(normal2)                                   #& 地图点指向第二个相机光心的向量距离
            cosParallax = np.dot(normal1_tran, normal2)/(dist1*dist2)   #& 视角差

            # 计算重投影误差
            z1_inv = float(1.0/p3dC1[2])                                #& 地图点的z坐标的倒数
            img1_x = fx*p3dC1[0]*z1_inv+cx                              #& 地图点投影到图像上
            img1_y = fy*p3dC1[1]*z1_inv+cy
            kp1_x = point[-2] 
            kp1_y = point[-1]
            squareError1 = (img1_x-kp1_x)*(img1_x-kp1_x)+(img1_y-kp1_y)*(img1_y-kp1_y)
            
            points3d_param.append([float(cosParallax), float(squareError1)])
            points3d_param_array = np.array(points3d_param)


        # 过滤条件
        filter_x = abs(point_array[:, 0]) < 200                          #& 滤去3D点x方向的异常值
        filter_y = abs(point_array[:, 1]) < 200                          #& 滤去3D点y方向的异常值
        filter_z = point_array[:, 2] > 0                                 #& 判断3D点是否在两个摄像头前方 z > 0
        filter_parallax = points3d_param_array[:, 0] < 0.999999          #& 视角差过滤
        #filter_squareError = points3d_param_array[:, 1] < 2.0*1.0*1.0   #& 重投影误差过滤
        screen_condition = filter_x & \
                            filter_y & \
                            filter_z & \
                            filter_parallax
                            #filter_squareError
                            #& 删选条件
        return screen_condition, points3d1, points3d2

    def _bundle_adjustment(self, pts1, pts2, R, t):
        print("test")

    def _add_semantic_label(self, flag, kps):
        print("test")

    # 反向投影消失点
    def _project_vanish_point(self, R, t, vanish_point):
        """
        @function: 将二维像素点反向投影到对应的三维世界坐标系的位置 
        @param:    R, t, vanish_point
        @return:   pts4d 
        """
        R = np.asmatrix(R).I                        # 旋转矩阵
        t_I = np.asmatrix(t).I                      # 平移矩阵(向量)
        C = np.hstack((R, t_I))                       
        P = np.asmatrix(self.K) * np.asmatrix(C)    # 成像矩阵

        pts3d = np.asmatrix([vanish_point[0], vanish_point[1], 1]).T
        pts4d = np.asmatrix(P).I * pts3d
        #pts4d = pts4d.transpose()

        self.vanish_points.append( [pts4d[0][0] * self.world_scale + self.camera_xyz[0],
                                    pts4d[1][0] * self.world_scale + self.camera_xyz[1],
                                    pts4d[2][0] * self.world_scale + self.camera_xyz[2]])
        
        self.vanish_points = (np.array(self.vanish_points))
        self.vanish_points = np.asmatrix(self.vanish_points)
        #print(pts4d[0])

        self.camera_xyz = self.camera_xyz + t_I
        #print(self.camera_xyz)
        #print(self.vanish_points)

        return self.vanish_points

def _save_data(filePath, idx, time):
    with open(filePath, 'a', encoding = 'utf-8') as f:
        f.writelines(str(idx) + ' ' + str(time) + '\n')

def draw_trajectory(R, t):
    traj_image = np.zeros((600, 600, 3), np.uint8)
    R_f = R
    t_f = t
    x = int(t_f[0]) + 100
    y = int(t_f[1]) + 300
    cv2.circle(traj_image, (x, y), 1, (0,255,0), 2)
    cv2.imshow("trajectory", traj_image)



def process_frame(frame, vanish_point):

    slam = Slam(width = 1920,
                height = 1080,
                fov = 270,
                matches_ratio = 0.75)
    # 提取当前帧的角点和描述子特征
    #t1 = cv2.getTickCount()
    frame.curr_kps, frame.curr_des = slam._extract_points(frame)
    #print(frame.curr_kps)
    #t2 = cv2.getTickCount()
    #time = (t2-t1)/cv2.getTickFrequency()
    #print("time : %s ms"%(time*1000)) 

    # 将角点位置和描述子通过类的属性传递给下一帧作为上一帧的角点信息
    Frame.last_kps, Frame.last_des = frame.curr_kps, frame.curr_des

    if frame.idx == 1:
        # 设置第一帧为初始帧，并以相机坐标系为世界坐标系
        frame.curr_pose = np.eye(4)  # 4x4矩阵
        map_pts4d = [[0,0,0,1]]      # 原点为[0, 0, 0],1表示颜色
        vanish_pts4d = [[0,0,0,1]] 
    else:
        #* 角点配准, 用RANSAC过滤掉一些噪声
        match_kps = slam._match_points(frame)
        #print(match_kps)
        fliePath = "./data/exp4.txt"
        #_save_data(fliePath, frame.idx, time)
        #print("frame: {}, curr_des: {}, last_des: {}, match_kps: {}, time: {}ms".
            #format(frame.idx, len(frame.curr_des), len(frame.last_des), len(match_kps), time))
        #* 使用八点法计算本质矩阵
        essential_matrix = slam._calculate_essential_matrix(match_kps)
        #print("---------------- Essential Matrix ----------------")
        #print(essential_matrix)
        #* 利用本质矩阵分解出相机的位姿Rt
        Rt, R, t = slam._extract_Rt(essential_matrix)
        #print("Rt:{}\n R:{}\n t:{}\n".format(Rt, R, t))
        #* 计算出当前帧相对于初始帧的相机位姿
        frame.curr_pose = np.dot(Rt, frame.last_pose) # 矩阵乘法
        #print(frame.curr_pose)
        #* 三角测量获得地图点
        map_pts4d = slam._triangulate(frame.last_kps, frame.curr_kps, frame.last_pose, frame.curr_pose)
        #print(frame.last_kps)
        #* 筛选地图点
        #print(len(map_pts4d))
        good_map_pt4d, points3d1, points3d2 = slam._check_points(frame.last_kps, frame.curr_kps, R, t, map_pts4d)
        map_pts4d = map_pts4d[good_map_pt4d]
        # TODO: g2o 后端优化
        #* 画出特征点
        slam._draw_points(frame)
        #* 画轨迹图
        draw_trajectory(R, t)
        #print(vanish_point)
        vanish_pts4d = slam._project_vanish_point(R, t, vanish_point)
        #print(vanish_pts4d)
    map.add_observation(frame.curr_pose, map_pts4d, vanish_pts4d)     # 将当前的pose和点云放入地图中
    # 将当前帧的pose信息存储为下一帧的 last_pose 信息
    Frame.last_pose = frame.curr_pose
    return frame


if __name__ == "__main__":
    speed = 30
    map = Map(1024, 768)    # 构建地图
    cap = cv2.VideoCapture("/home/lxj/桌面/lane_line_sfm/video/video.mp4")
    #print(cap.get(3))
    #print(cap.get(4))
    while cap.isOpened() :
        ret, image = cap.read()
        frame = Frame(image)

        if ret:
            laneline = Laneline(gaussian_ksize = 5, 
                                gaussian_sigmax = 1, 
                                canny_threshold1 = 50, 
                                canny_threshold2 = 100,
                                slope_diff = 0.2,
                                line_threshold = 15,
                                linelength_min = 60,
                                linegap_max = 20)
            vanish_point = laneline._lane_line_detection(frame.image)
            frame = process_frame(frame, vanish_point)
        else:
            break

        cv2.imshow("slam", frame.image)
        map.display()
        key = cv2.waitKey(speed)
        if key == ord('q'): 
            break
