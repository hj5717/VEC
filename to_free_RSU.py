# 选择所有服务器中空闲的那个
import numpy as np


def free_rsu(servers):
    free_RSU = np.argmin(servers)
    servers_road = 1

    if 20 <= free_RSU <= 39:
        servers_road = 2
        free_RSU -= 20
    elif 40 <= free_RSU <=57:
        servers_road = 3
        free_RSU -= 40
    elif 58 <= free_RSU <= 75:
        servers_road = 4
        free_RSU -= 58

    return servers_road, free_RSU


# print(free_rsu([1, 1, 0, 0, 1]))
