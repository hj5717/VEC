"""
    设计为‘＃’型道路，每条道路为双向行驶
    每条道路可以分为5个路段，城外，交叉口，城内，交叉口，城外
    不同路段速度不同，分别为城外：72km/h，城内：36km/h，交叉口：18km/h  即20m/s，10m/s，5m/s

    每个单向道路长度为8km，每400米一段，分段分别为城外（2400m 6段）交叉口（400m 1段）城内（2400m 6段）交叉口（400m 1段）城外（2400m 6段）
    其中交叉口路段为多个路段共享

    每个RSU服务器的服务范围为边长为400米的正方形
"""
import numpy as np
import math
import torch


# 定义道路类
class Roads:
    def __init__(self):
        self.speed = np.zeros([8, 20])
        for i in range(8):
            for j in range(20):
                if 0 <= j <= 5 or 14 <= j <= 19:
                    self.speed[i, j] = 20
                elif 7 <= j <= 12:
                    self.speed[i, j] = 10
                elif j == 6 or j == 13:
                    self.speed[i, j] = 5

        # 记录道路类型
        self.index = np.zeros(20)
        for i in range(20):
            if 0 <= i <= 5 or 14 <= i <= 19:
                self.index[i] = 0
            elif 7 <= i <= 12:
                self.index[i] = 1
            elif i == 6 or i == 13:
                self.index[i] = 2

        self.length = 8000

    def get_speed(self, road, position):
        speed_index = math.floor(position / 400)
        return self.speed[road - 1, speed_index]

    def get_place(self, position):
        road_index = math.floor(position / 400)
        return self.index[road_index]


# roads = Roads()
# print(roads.speed)
# print(roads.index)
