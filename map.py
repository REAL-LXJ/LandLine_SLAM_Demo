from turtle import color
from matplotlib import colors
from matplotlib.pyplot import flag
import pangolin
import OpenGL.GL as gl
import numpy as np
from multiprocessing import Process, Queue

def _save_data(filePath, x, y):
    with open(filePath, 'a', encoding = 'utf-8') as f:
        f.writelines(str(x) + ' ' + str(y) + '\n')


# 构建地图，显示角点的点云和相机的位姿
class Map:
    def __init__(self, width, height):
        self.width  = width
        self.height = height
        self.poses  = []                    # 相机位姿
        self.points = []                    # 地图点集
        self.vanish_points = []             # 消失点集
        self.state = None
        self.q = Queue()

        p = Process(target=self.viewer_thread, args=(self.q,))
        p.daemon = True
        p.start()

    # 添加观测
    def add_observation(self, pose, points, vanish_points):
        self.poses.append(pose)
        self.vanish_points = vanish_points
        #print(self.vanish_points)
        for point in points:
            self.points.append(point)
            #print(point)
            

    def viewer_init(self):
        #& 1.创建视窗
        pangolin.CreateWindowAndBind('Map', self.width, self.height)
        #& 2.启动深度测试
        gl.glEnable(gl.GL_DEPTH_TEST)

        #& 3.创建一个观察相机
        # ProjectMatrix(int h, int w, int fu, int fv, int cu, int cv, int znear, int zfar) 
        # 参数依次为观察相机的图像高度、宽度、4个内参以及最近和最远视距
        # ModelViewLookAt(double x, double y, double z,double lx, double ly, double lz, AxisDirection Up)
        # 参数依次为相机所在的位置，以及相机所看的视点位置(一般会设置在原点)
        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(self.width, self.height, 420, 420, self.width//2, self.height//2, 0.1, 1000),
            pangolin.ModelViewLookAt(  -8, 1,-8,
                                        0, 0, 0,
                                        0, 1, 0))
        # setBounds()函数前四个参数依次表示视图在视窗中的范围（下、上、左、右），可以采用相对坐标（0~1）以及绝对坐标（使用Attach对象）
        self.handler = pangolin.Handler3D(self.scam) # 交互相机视图句柄

        
        #& 4.在窗口中创建交互式视图
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -self.width/self.height)
        self.dcam.SetHandler(self.handler)

        self.panel = pangolin.CreatePanel('ui')
        self.panel.SetBounds(0.0, 1.0, 0.0, 120/self.height)

        points = pangolin.VarBool('ui.Show Points', value=False, toggle=False)
        poses = pangolin.VarBool('ui.Show Pose', value=False, toggle=False)
    
    def viewer_thread(self, q):
        self.viewer_init()
        while True:
            self.viewer_refresh(q)

    def viewer_refresh(self, q):
        if self.state is None or not q.empty():
            self.state = q.get()

        # 清空颜色和深度缓存
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        self.dcam.Activate(self.scam)

        # 绘制位姿
        gl.glColor3f(0.0, 1.0, 0.0)
        pangolin.DrawCameras(self.state[0])

        # 绘制地图点
        # N*4
        gl.glPointSize(2)
        colors = np.zeros((len(self.state[1]), 3))
        for i in range(len(self.state[1])):
            depth = self.state[1][i][2]
            if depth < 20:#车道线
                colors[i] = [1.0, 0.0, 0.0]
            elif depth > 20 and depth < 60:
                colors[i] = [0.0, 1.0, 0.0]
            elif depth > 60 and depth < 100:
                colors[i] = [0.0, 0.0, 1.0]
            else:
                colors[i] = [0.0, 1.0, 1.0]
            #print(depth)
            #fliePath = "./data/depth.txt"
            #_save_data(fliePath, i, depth)
        pangolin.DrawPoints(self.state[1], colors)

        # 绘制消失点
        gl.glPointSize(10)
        gl.glColor3f(0.0, 0.0, 1.0)
        pangolin.DrawPoints(self.state[2])


        '''
        trajectory = [[0, -6, 6]]
        for i in range(300):
            trajectory.append(trajectory[-1] + np.random.random(3)-0.5)
        trajectory = np.array(trajectory)
        print(trajectory)
        '''

        # 绘制车道线
        gl.glLineWidth(1)
        gl.glColor3f(0.0, 0.0, 0.0)
        #pangolin.DrawLine(trajectory)   # consecutive

        # 绘制车辆框
        #box_poses = [np.identity(4)]
        sizes = [[0.5, 0.5, 0.5]]
        car_box1 = [(  [[1., 2., 0., 0.],
                        [0., 1., 0., 4.2],
                        [0., 0., 1., 6.],
                        [0., 0., 0., 0.8]])]
        car_box2 = [([  [1., 2., 0., 0.],
                        [0., 1., 0., 4.2],
                        [0., 0., 1., 6.],
                        [0., 0., 0., 1.2]])]
        car_box3 = [([  [1., 2., 0., 0.],
                        [0., 1., 0., 4.2],
                        [0., 0., 1., 6.],
                        [0., 0., 0., 1.6]])]
        
        gl.glLineWidth(3)
        gl.glColor3f(1.0, 0.0, 1.0)
        #pangolin.DrawBoxes(car_box1, sizes)

        gl.glLineWidth(3)
        gl.glColor3f(0.0, 0.2, 0.5)
        #pangolin.DrawBoxes(car_box2, sizes)

        gl.glLineWidth(3)
        gl.glColor3f(0.0, 1.0, 0.8)
        #pangolin.DrawBoxes(car_box3, sizes)

        pangolin.FinishFrame()

    def display(self):
        poses = np.array(self.poses)
        points = np.array(self.points)
        vanish_points = np.array(self.vanish_points)
        self.q.put((poses, points, vanish_points))