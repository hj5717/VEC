import numpy as np

'''
    共有三种任务
        导航类任务：固定每秒生成一次
        避障类任务：不同路段生成概率不同，每秒判定一次
        娱乐类任务：固定生成概率，每秒判定一次
    
    期望延迟影响满意度
    最大延迟影响任务卸载成功率
'''
# 三种任务类型
#   导航类任务：
#       判定周期：1s
#       产生概率：100%
#       TODO 期望延迟：0.5s
#       TODO 最大延迟：1s
#       TODO 上传/下载额外数据：20/20kB
#       TODO 任务大小：300kb
#       TODO RSU/云虚拟机的占用率：6/3
#   危险判定类任务：
#       判定周期：1s
#       产生概率：城外30% 城内60% 交叉口90%
#       TODO 期望延迟：0.5s
#       TODO 最大延迟：1s
#       TODO 上传/下载额外数据：40/20kB
#       TODO 任务大小：1000kb
#       TODO RSU/云虚拟机的占用率：20/10
#   娱乐类任务：
#       判定周期：1s
#       产生概率：15%
#       TODO 期望延迟：1.5s
#       TODO 最大延迟：3
#       TODO 上传/下载额外数据：20/80kB
#       TODO 任务大小：2000kb
#       TODO RSU/云虚拟机的占用率：40/20


# 导航任务
def Navigation_work():
    return 300


def Obstacle_Avoidance_work(place):
    # 用于判定占用率的参数，不一定用得到
    is_product = 0

    x = [0, 1000]
    p = []
    if place == 0:
        p = [0.7, 0.3]
    elif place == 1:
        p = [0.4, 0.6]
    elif place == 2:
        p = [0.1, 0.9]

    work = np.random.choice(x, p=p)
    if work > 0:
        is_product = 1

    return work, is_product


def Entertainment_work():
    is_product = 0

    x = [0, 2000]
    p = [0.85, 0.15]

    work = np.random.choice(x, p=p)
    if work > 0:
        is_product = 1

    return work, is_product


def all_work(place):
    work1, work2, work3 = Navigation_work(), Obstacle_Avoidance_work(place), Entertainment_work()

    work = work1 + work2[0] + work3[0]
    # occupy_RSU = o1*6 + o2*20 + o3*40
    # occupy_cloud = o1*1.6 + o2*4 + o3*8
    # return work, occupy_RSU, occupy_cloud
    return work


# for i in range(100):
#     print(all_work(0))
