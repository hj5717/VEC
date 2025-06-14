"""
车辆：
    决策主体，按照一定逻辑进行移动和产生任务
    拥有一定计算能力
    为保证车辆数量，设定每驶出一辆车，就进入一辆车
VEC服务器：
    为边长400米的车辆提供计算支持
    拥有较大的计算能力，与车辆和云端通过无线网络进行连接
    VEC服务器之间通过有线网络进行连接，拥有极强的传输能力
云服务器：
    为车辆和VEC服务器提供远程计算支持
    拥有极强的计算能力，但是传输时延较大
"""
from road import Roads
from data_product import all_work
import math

road = Roads()


# 定义
class vehicle:
    def __init__(self, pos, road_, index):
        # 三种动作空间
        self.action_space = ['0', '1', '2']
        self.n_actions = len(self.action_space)
        # TODO 状态空间暂定
        self.n_features = 4

        # 位置、速度、路段和属于哪一条路
        self.pos = pos
        self.speed = road.get_speed(road_, pos)
        self.place = road.get_place(pos)
        self.road_ = road_

        self.work_size = 0
        self.line = 0
        # TODO
        self.power = 2000
        # 方向指示器，为0表示正向行驶，为1表示反向行驶
        self.index = index

    def forward(self):
        # TODO 先尝试一下不移动会怎么样，看看有没有效果
        if self.index == 0:
            self.pos += self.speed
            if self.pos >= 8000:
                self.pos = self.pos % 8000
            self.speed = road.get_speed(self.road_, self.pos)
            self.place = road.get_place(self.pos)
        elif self.index == 1:
            self.pos -= self.speed
            if self.pos < 0:
                self.pos = self.pos + 8000
            self.speed = road.get_speed(self.road_, self.pos)
            self.place = road.get_place(self.pos)
        # 处理任务
        self.line -= self.power
        if self.line < 0:
            self.line = 0

    def product_data(self):
        self.work_size = all_work(self.place)
        return self.work_size

    def get_server(self):
        server_choose = math.floor(self.pos / 400)
        return server_choose

    def pop_line(self, work_size):
        self.line += work_size
        return self.line

    def reset(self, pos):
        self.line = 0

        self.pos = pos
        self.speed = road.get_speed(self.road_, self.pos)
        self.place = road.get_place(self.pos)


# VEC服务器类
class MecServer:
    def __init__(self, ID):
        self.ID = ID
        self.line = 0
        self.power = 10000

    def pop_line(self, work_size):
        self.line += work_size
        return self.line

    def handle_works(self):
        self.line -= self.power
        if self.line < 0:
            self.line = 0

    def reset(self):
        self.line = 0


# 云服务器类
class cloudServer:
    def __init__(self):
        self.line = 0
        self.power = 280000

    def pop_line(self, work_size):
        self.line += work_size
        return self.line

    def handle_works(self):
        self.line -= self.power
        if self.line < 0:
            self.line = 0

    def reset(self):
        self.line = 0
