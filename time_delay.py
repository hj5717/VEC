import numpy as np

"""
    任务传输设定
    1.车辆本身拥有一定处理能力，能处理较小的任务
    2.车辆与RSU服务器通过WLAN接口相连，使用IEEE802.11协议进行V2R通信
    3.RSU服务器之间通过城域网（MAN）连接，通过任务迁移来共享计算能力
        当车辆在卸载任务执行之前离开了服务RSU的范围，则结果将通过其它RSU以多跳方式传输给相关车辆
    4.车辆与云端通过蜂窝网络相连
    5.车辆到云端的传输有固定传输时延

    参考相关文献后，对传输速率做以下设定：
        车辆和rsu之间：10 Mbps
        车辆和云之间：20 Mbps
        RSU之间：通过有线网络连接，1000Mps
"""

'''
    时延计算设定
    使用抗信噪比公式，来计算任务传输速率
    若选择本地处理
        排队时延+处理时延
    若选择卸载到RSU服务器
        上传时延+排队时延+处理时延+结果返回时延
    若选择直接卸载到云服务器
        上传时延+传输时延+排队时延+处理时延+传输时延+结果返回时延
'''
# 任务大小和返回结果大小的比值
D_d = 0.1
# 车辆的信号发射功率
P = 10
# 信道增益,综合取值后得一
O = 1


# 计算抗信噪比
def sinr(n):
    U = abs(np.random.normal(1, 1))
    if n == 1:
        sinr_ = 1
    else:
        sinr_ = (P * O) / ((n - 1) * P * O + U)
    return sinr_


def cal_time_local(l, d, f):
    t = (l + d) / f
    return t


def cal_time_RSU(l, d, f, n):
    r = 300000 * np.log2(1 + sinr(n))
    # 上传时延+处理时延+结果返回时延+车辆到RSU服务器传输时延
    t = d / r + (l + d) / f + 0.01
    return t


# 将任务卸载到空闲的服务器处理
def cal_time_free_RSU(l, d, f, n):
    r = 300000 * np.log2(1 + sinr(n))
    # 上传时延+处理时延+结果返回时延+车辆到RSU服务器传输时延传输时延+RSU服务器之间的传输时延
    t = d / r + (l + d) / f + 0.015
    return t


def cal_time_cloud(l, d, f, n):
    r = 600000 * np.log2(1 + sinr(n))
    # 上传时延+处理时延+结果返回时延
    t = d / r + (l + d) / f + 0.02
    return t
