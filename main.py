from vec_env import MecServer, cloudServer, vehicle
from SAC_brain import SAC, ReplayBuffer
from time_delay import cal_time_local, cal_time_RSU, cal_time_cloud, cal_time_free_RSU
from to_free_RSU import free_rsu
import torch
import numpy as np
import pandas as pd

"""
    1.考虑了更接近真实场景的复杂路况，研究具有更好的适用性
    2.设计了一种具有共享VEC服务器池的任务卸载策略
    3.引入了关键任务优先机制，进一步优化任务卸载，提高用户满意度
"""

"""
    以边缘设备为卸载决策中心
    动作空间：[本地计算，所属服务器计算，共享VEC服务器池计算，云端计算]
    状态空间：[任务大小，本地队列长度，所属服务器队列长度，最短VEC服务器队列长度，云端长度]
    奖励机制：是否为时延最小动作
"""

"""
    共享VEC服务器池：
        VEC服务器之间通过有线链路连接，传输效率较快
        设定共享VEC服务器池，VEC服务器之间能够共享计算能力
        当任务所属VEC服务器负担较重时，可以通过共享机制将任务卸载到共享池中空闲的服务器进行处理
    
    关键任务优先机制：
        在几类任务中，如避障类任务等，为关键任务，需要优先执行
        调整关键任务的状态空间，使用关键任务类的队列作为状态
        当任务卸载到某一设备上后，将其插入关键任务类队列中优先处理
"""

'''--------训练设置--------'''
# 测试次数
n_test = 1
n_train = 5
# 训练轮次
epochs = 300
# 每个轮次的时间片数量
n_timeslot = 100
'''--------神经网络设置--------'''
capacity = 500  # 经验池容量
min_size = 200  # 经验池训练容量
batch_size = 64
n_hiddens = 64
actor_lr = 1e-3  # 策略网络学习率
critic_lr = 1e-2  # 价值网络学习率
alpha_lr = 1e-2  # 课训练变量的学习率
target_entropy = -1
tau = 0.005  # 软更新参数
gamma = 0.9  # 折扣因子
device = torch.device('cuda') if torch.cuda.is_available() \
    else torch.device('cpu')
buffer = ReplayBuffer(capacity=capacity)
'''--------设备数量--------'''
# 共八条单向道路，四条双向道路，每条双向道路20台设备，再减去交叉路口的四个，共76个
n_servers = [20, 20, 18, 18]
# 设定车辆数量
n_vehicle = 800
'''--------道路数量--------'''
# 初始化一个道路类，使用其中的信息来创建设备类
n_road = 8
every_vehicle_of_road = n_vehicle / n_road

t_epochs = np.zeros([n_test, epochs])

for test in range(n_test):
    print("test now: " + str(test + 1))

    # 实例化边缘服务器
    # 每条双向道路上实例化一组
    servers_1 = []
    servers_2 = []
    servers_3 = []
    servers_4 = []
    for i in range(n_servers[0]):
        servers_1.append(MecServer(i))
    for i in range(n_servers[1]):
        servers_2.append(MecServer(i))
    for i in range(n_servers[2]):
        servers_3.append(MecServer(i))
    for i in range(n_servers[3]):
        servers_4.append(MecServer(i))

    # print('各个道路上的服务器长度为：', len(servers_1), len(servers_2), len(servers_3), len(servers_4))

    # 实例化云服务器
    cloud = cloudServer()

    # 实例化车辆
    n_road_vehicle = int(n_vehicle / n_road)
    # print('每条路的车辆数为', n_road_vehicle)
    vehicles_1 = []
    vehicles_2 = []
    vehicles_3 = []
    vehicles_4 = []
    vehicles_5 = []
    vehicles_6 = []
    vehicles_7 = []
    vehicles_8 = []
    for ID in range(n_road_vehicle):
        pos = ID * (8000 / n_road_vehicle)
        vehicles_1.append(vehicle(pos, 1, 0))
        vehicles_2.append(vehicle(pos, 2, 1))
        vehicles_3.append(vehicle(pos, 3, 0))
        vehicles_4.append(vehicle(pos, 4, 1))
        vehicles_5.append(vehicle(pos, 5, 0))
        vehicles_6.append(vehicle(pos, 6, 1))
        vehicles_7.append(vehicle(pos, 7, 0))
        vehicles_8.append(vehicle(pos, 8, 1))

    # print('每条单向道路上的车辆数量为', len(vehicles_1))

    # 工具环境
    env = vehicle(0, 0, 0)

    # 为每个车辆创建一个智能体
    multi_agent = []
    multi_replay_buffer = []
    for ID in range(n_vehicle):
        multi_agent.append(SAC(n_states=env.n_features, n_hiddens=n_hiddens, n_actions=env.n_actions,
                               actor_lr=actor_lr, critic_lr=critic_lr, alpha_lr=alpha_lr,
                               target_entropy=target_entropy,
                               tau=tau, gamma=gamma, device=device))
        multi_replay_buffer.append(buffer)
    # print('智能体的数量为：', len(multi_agent))

    for epoch in range(epochs):
        print("epoch now: " + str(epoch + 1))

        # 存储一次epochs内所有数据，用于阶段式的学习
        s = np.zeros([n_timeslot, env.n_features * n_vehicle])
        s_ = np.zeros([n_timeslot, env.n_features * n_vehicle])
        r = np.zeros([n_timeslot, n_vehicle])
        action = np.zeros([n_timeslot, n_vehicle])

        t_timeslots = np.zeros(n_timeslot)

        # 每个训练轮次开始时，初始化云端、RSU服务器和车辆

        cloud.reset()
        for i in range(n_servers[0]):
            servers_1[i].reset()
        for i in range(n_servers[1]):
            servers_2[i].reset()
        for i in range(n_servers[2]):
            servers_3[i].reset()
        for i in range(n_servers[3]):
            servers_4[i].reset()
        for ID in range(n_road_vehicle):
            pos = ID * (8000 / n_road_vehicle)
            vehicles_1[ID].reset(pos)
            vehicles_2[ID].reset(pos)
            vehicles_3[ID].reset(pos)
            vehicles_4[ID].reset(pos)
            vehicles_5[ID].reset(pos)
            vehicles_6[ID].reset(pos)
            vehicles_7[ID].reset(pos)
            vehicles_8[ID].reset(pos)

        '''
            运行所有时间片后，储存数据
        '''
        for timeslot in range(n_timeslot):
            # print("timeslot now: " + str(timeslot + 1))
            # # 记录一个时间片内每个车辆所属的RSU边缘服务器
            # all_offload_server = np.zeros([n_road, n_road_vehicle])

            # 记录每个时间片中，并行传输至某服务器的任务数量，以计算传输时延
            n_to_server_1 = np.zeros(n_servers[0])
            n_to_server_2 = np.zeros(n_servers[1])
            n_to_server_3 = np.zeros(n_servers[2])
            n_to_server_4 = np.zeros(n_servers[3])

            # print('各个道路上卸载至某服务器的任务数量长度为：', len(n_to_server_1), len(n_to_server_2),
            #       len(n_to_server_3), len(n_to_server_4))

            n_to_cloud = 0

            # 记录车辆所属的服务器
            server_choose = 0

            # 记录一个时间片内所有车辆所产生任务大小
            all_works = np.zeros([n_road, n_road_vehicle])
            """
            第一次循环：
                获取观测值
                使用观测值获取动作选择和动作的概率
                获得动作的二元化表示
                统计同一轮中选择卸载到所属服务器和云端的任务数量
            """
            for ID in range(n_vehicle):
                road = int((ID / every_vehicle_of_road) + 1)
                real_ID = ID
                ID = int(ID % every_vehicle_of_road)
                # print('这辆车所属的road为:', road, 'ID为:', ID)
                work_size = 0
                if road == 1:
                    server_choose = vehicles_1[ID].get_server()
                    work_size = vehicles_1[ID].product_data()
                    # print('这辆车所属的服务器为:', server_choose, '产生的任务大小为:', work_size)
                    # TODO 状态设定为任务大小，本地队列长度、所属服务器的队列长度，云服务器的队列长度
                    s[timeslot, real_ID * env.n_features:real_ID * env.n_features + env.n_features] = \
                        [work_size, vehicles_1[ID].line, servers_1[server_choose].line, cloud.line]
                    # print('初始状态为：',
                    #       s[timeslot, real_ID * env.n_features:real_ID * env.n_features + env.n_features])

                    the_action = (multi_agent[real_ID].take_action(s[timeslot, real_ID * env.n_features:
                                                                         real_ID * env.n_features + env.n_features]))
                    action[timeslot, real_ID] = int(the_action)
                    # 动作0：本地计算 动作1：卸载到边缘服务器 动作2：卸载到云端
                    # print('第一次选择的动作为：', action[timeslot, real_ID])
                    if action[timeslot, real_ID] == 1:
                        n_to_server_1[server_choose] += 1
                    elif action[timeslot, real_ID] == 2:
                        n_to_cloud += 1
                elif road == 2:
                    server_choose = vehicles_2[ID].get_server()
                    work_size = vehicles_2[ID].product_data()
                    s[timeslot, real_ID * env.n_features:real_ID * env.n_features + env.n_features] = \
                        [work_size, vehicles_2[ID].line, servers_1[server_choose].line, cloud.line]
                    the_action = (multi_agent[real_ID].take_action(s[timeslot, real_ID * env.n_features:
                                                                         real_ID * env.n_features + env.n_features]))
                    action[timeslot, real_ID] = int(the_action)
                    if action[timeslot, real_ID] == 1:
                        n_to_server_1[server_choose] += 1
                    elif action[timeslot, real_ID] == 2:
                        n_to_cloud += 1
                elif road == 3:
                    server_choose = vehicles_3[ID].get_server()
                    work_size = vehicles_3[ID].product_data()
                    s[timeslot, real_ID * env.n_features:real_ID * env.n_features + env.n_features] = \
                        [work_size, vehicles_3[ID].line, servers_2[server_choose].line, cloud.line]
                    the_action = (multi_agent[real_ID].take_action(s[timeslot, real_ID * env.n_features:
                                                                         real_ID * env.n_features + env.n_features]))
                    action[timeslot, real_ID] = int(the_action)
                    if action[timeslot, real_ID] == 1:
                        n_to_server_2[server_choose] += 1
                    elif action[timeslot, real_ID] == 2:
                        n_to_cloud += 1
                elif road == 4:
                    server_choose = vehicles_4[ID].get_server()
                    work_size = vehicles_4[ID].product_data()
                    s[timeslot, real_ID * env.n_features:real_ID * env.n_features + env.n_features] = \
                        [work_size, vehicles_4[ID].line, servers_2[server_choose].line, cloud.line]
                    the_action = (multi_agent[real_ID].take_action(s[timeslot, real_ID * env.n_features:
                                                                         real_ID * env.n_features + env.n_features]))
                    action[timeslot, real_ID] = int(the_action)
                    if action[timeslot, real_ID] == 1:
                        n_to_server_2[server_choose] += 1
                    elif action[timeslot, real_ID] == 2:
                        n_to_cloud += 1
                elif road == 5:
                    server_choose = vehicles_5[ID].get_server()
                    work_size = vehicles_5[ID].product_data()
                    # 当所属区域为交叉口时，两条双向道路共享一个VEC服务器，因此需要做一些处理
                    if server_choose == 6:
                        s[timeslot, real_ID * env.n_features:real_ID * env.n_features + env.n_features] = \
                            [work_size, vehicles_5[ID].line, servers_1[6].line, cloud.line]
                        the_action = (multi_agent[real_ID].take_action(s[timeslot, real_ID * env.n_features:
                                                                             real_ID * env.n_features + env.n_features]))
                        action[timeslot, real_ID] = int(the_action)
                        if action[timeslot, real_ID] == 1:
                            n_to_server_1[6] += 1
                        elif action[timeslot, real_ID] == 2:
                            n_to_cloud += 1
                    elif server_choose == 13:
                        s[timeslot, real_ID * env.n_features:real_ID * env.n_features + env.n_features] = \
                            [work_size, vehicles_5[ID].line, servers_2[6].line, cloud.line]
                        the_action = (multi_agent[real_ID].take_action(s[timeslot, real_ID * env.n_features:
                                                                             real_ID * env.n_features + env.n_features]))
                        action[timeslot, real_ID] = int(the_action)
                        if action[timeslot, real_ID] == 1:
                            n_to_server_2[6] += 1
                        elif action[timeslot, real_ID] == 2:
                            n_to_cloud += 1
                    elif 6 < server_choose < 13:
                        server_choose -= 1
                        s[timeslot, real_ID * env.n_features:real_ID * env.n_features + env.n_features] = \
                            [work_size, vehicles_5[ID].line, servers_3[server_choose].line, cloud.line]
                        the_action = (multi_agent[real_ID].take_action(s[timeslot, real_ID * env.n_features:
                                                                             real_ID * env.n_features + env.n_features]))
                        action[timeslot, real_ID] = int(the_action)
                        if action[timeslot, real_ID] == 1:
                            n_to_server_3[server_choose] += 1
                        elif action[timeslot, real_ID] == 2:
                            n_to_cloud += 1
                    elif server_choose > 13:
                        server_choose -= 2
                        s[timeslot, real_ID * env.n_features:real_ID * env.n_features + env.n_features] = \
                            [work_size, vehicles_5[ID].line, servers_3[server_choose].line, cloud.line]
                        the_action = (multi_agent[real_ID].take_action(s[timeslot, real_ID * env.n_features:
                                                                             real_ID * env.n_features + env.n_features]))
                        action[timeslot, real_ID] = int(the_action)
                        if action[timeslot, real_ID] == 1:
                            n_to_server_3[server_choose] += 1
                        elif action[timeslot, real_ID] == 2:
                            n_to_cloud += 1
                    else:
                        s[timeslot, real_ID * env.n_features:real_ID * env.n_features + env.n_features] = \
                            [work_size, vehicles_5[ID].line, servers_3[server_choose].line, cloud.line]
                        the_action = (multi_agent[real_ID].
                                                     take_action(s[timeslot, real_ID * env.n_features:
                                                                             real_ID * env.n_features + env.n_features]))
                        action[timeslot, real_ID] = int(the_action)
                        if action[timeslot, real_ID] == 1:
                            n_to_server_3[server_choose] += 1
                        elif action[timeslot, real_ID] == 2:
                            n_to_cloud += 1
                elif road == 6:
                    server_choose = vehicles_6[ID].get_server()
                    work_size = vehicles_6[ID].product_data()
                    if server_choose == 6:
                        s[timeslot, real_ID * env.n_features:real_ID * env.n_features + env.n_features] = \
                            [work_size, vehicles_6[ID].line, servers_1[6].line, cloud.line]
                        the_action = (multi_agent[real_ID].take_action(s[timeslot, real_ID * env.n_features:
                                                                             real_ID * env.n_features + env.n_features]))
                        action[timeslot, real_ID] = int(the_action)
                        if action[timeslot, real_ID] == 1:
                            n_to_server_1[6] += 1
                        elif action[timeslot, real_ID] == 2:
                            n_to_cloud += 1
                    elif server_choose == 13:
                        s[timeslot, real_ID * env.n_features:real_ID * env.n_features + env.n_features] = \
                            [work_size, vehicles_6[ID].line, servers_2[6].line, cloud.line]
                        the_action = (multi_agent[real_ID].take_action(s[timeslot, real_ID * env.n_features:
                                                                             real_ID * env.n_features + env.n_features]))
                        action[timeslot, real_ID] = int(the_action)
                        if action[timeslot, real_ID] == 1:
                            n_to_server_2[6] += 1
                        elif action[timeslot, real_ID] == 2:
                            n_to_cloud += 1
                    elif 6 < server_choose < 13:
                        server_choose -= 1
                        s[timeslot, real_ID * env.n_features:real_ID * env.n_features + env.n_features] = \
                            [work_size, vehicles_6[ID].line, servers_3[server_choose].line, cloud.line]
                        the_action = (multi_agent[real_ID].take_action(s[timeslot, real_ID * env.n_features:
                                                                             real_ID * env.n_features + env.n_features]))
                        action[timeslot, real_ID] = int(the_action)
                        if action[timeslot, real_ID] == 1:
                            n_to_server_3[server_choose] += 1
                        elif action[timeslot, real_ID] == 2:
                            n_to_cloud += 1
                    elif server_choose > 13:
                        server_choose -= 2
                        s[timeslot, real_ID * env.n_features:real_ID * env.n_features + env.n_features] = \
                            [work_size, vehicles_6[ID].line, servers_3[server_choose].line, cloud.line]
                        the_action = (multi_agent[real_ID].take_action(s[timeslot, real_ID * env.n_features:
                                                                             real_ID * env.n_features + env.n_features]))
                        action[timeslot, real_ID] = int(the_action)
                        if action[timeslot, real_ID] == 1:
                            n_to_server_3[server_choose] += 1
                        elif action[timeslot, real_ID] == 2:
                            n_to_cloud += 1
                    else:
                        s[timeslot, real_ID * env.n_features:real_ID * env.n_features + env.n_features] = \
                            [work_size, vehicles_6[ID].line, servers_3[server_choose].line, cloud.line]
                        the_action = (multi_agent[real_ID].take_action(s[timeslot, real_ID * env.n_features:
                                                                             real_ID * env.n_features + env.n_features]))
                        action[timeslot, real_ID] = int(the_action)
                        if action[timeslot, real_ID] == 1:
                            n_to_server_3[server_choose] += 1
                        elif action[timeslot, real_ID] == 2:
                            n_to_cloud += 1
                elif road == 7:
                    server_choose = vehicles_7[ID].get_server()
                    work_size = vehicles_7[ID].product_data()
                    if server_choose == 6:
                        s[timeslot, real_ID * env.n_features:real_ID * env.n_features + env.n_features] = \
                            [work_size, vehicles_7[ID].line, servers_1[13].line, cloud.line]
                        the_action = (multi_agent[real_ID].take_action(s[timeslot, real_ID * env.n_features:
                                                                             real_ID * env.n_features + env.n_features]))
                        action[timeslot, real_ID] = int(the_action)
                        if action[timeslot, real_ID] == 1:
                            n_to_server_1[13] += 1
                        elif action[timeslot, real_ID] == 2:
                            n_to_cloud += 1
                    elif server_choose == 13:
                        s[timeslot, real_ID * env.n_features:real_ID * env.n_features + env.n_features] = \
                            [work_size, vehicles_7[ID].line, servers_2[13].line, cloud.line]
                        the_action = (multi_agent[real_ID].take_action(s[timeslot, real_ID * env.n_features:
                                                                             real_ID * env.n_features + env.n_features]))
                        action[timeslot, real_ID] = int(the_action)
                        if action[timeslot, real_ID] == 1:
                            n_to_server_2[13] += 1
                        elif action[timeslot, real_ID] == 2:
                            n_to_cloud += 1
                    elif 6 < server_choose < 13:
                        server_choose -= 1
                        s[timeslot, real_ID * env.n_features:real_ID * env.n_features + env.n_features] = \
                            [work_size, vehicles_7[ID].line, servers_4[server_choose].line, cloud.line]
                        the_action = (multi_agent[real_ID].take_action(s[timeslot, real_ID * env.n_features:
                                                                             real_ID * env.n_features + env.n_features]))
                        action[timeslot, real_ID] = int(the_action)
                        if action[timeslot, real_ID] == 1:
                            n_to_server_4[server_choose] += 1
                        elif action[timeslot, real_ID] == 2:
                            n_to_cloud += 1
                    elif server_choose > 13:
                        server_choose -= 2
                        s[timeslot, real_ID * env.n_features:real_ID * env.n_features + env.n_features] = \
                            [work_size, vehicles_7[ID].line, servers_4[server_choose].line, cloud.line]
                        the_action = (multi_agent[real_ID].take_action(s[timeslot, real_ID * env.n_features:
                                                                             real_ID * env.n_features + env.n_features]))
                        action[timeslot, real_ID] = int(the_action)
                        if action[timeslot, real_ID] == 1:
                            n_to_server_4[server_choose] += 1
                        elif action[timeslot, real_ID] == 2:
                            n_to_cloud += 1
                    else:
                        s[timeslot, real_ID * env.n_features:real_ID * env.n_features + env.n_features] = \
                            [work_size, vehicles_7[ID].line, servers_4[server_choose].line, cloud.line]
                        the_action = (multi_agent[real_ID].take_action(s[timeslot, real_ID * env.n_features:
                                                                             real_ID * env.n_features + env.n_features]))
                        action[timeslot, real_ID] = int(the_action)
                        if action[timeslot, real_ID] == 1:
                            n_to_server_4[server_choose] += 1
                        elif action[timeslot, real_ID] == 2:
                            n_to_cloud += 1
                elif road == 8:
                    server_choose = vehicles_8[ID].get_server()
                    work_size = vehicles_8[ID].product_data()
                    if server_choose == 6:
                        s[timeslot, real_ID * env.n_features:real_ID * env.n_features + env.n_features] = \
                            [work_size, vehicles_8[ID].line, servers_1[13].line, cloud.line]
                        the_action = (multi_agent[real_ID].take_action(s[timeslot, real_ID * env.n_features:
                                                                             real_ID * env.n_features + env.n_features]))
                        action[timeslot, real_ID] = int(the_action)
                        if action[timeslot, real_ID] == 1:
                            n_to_server_1[13] += 1
                        elif action[timeslot, real_ID] == 2:
                            n_to_cloud += 1
                    elif server_choose == 13:
                        s[timeslot, real_ID * env.n_features:real_ID * env.n_features + env.n_features] = \
                            [work_size, vehicles_8[ID].line, servers_2[13].line, cloud.line]
                        the_action = (multi_agent[real_ID].take_action(s[timeslot, real_ID * env.n_features:
                                                                             real_ID * env.n_features + env.n_features]))
                        action[timeslot, real_ID] = int(the_action)
                        if action[timeslot, real_ID] == 1:
                            n_to_server_2[13] += 1
                        elif action[timeslot, real_ID] == 2:
                            n_to_cloud += 1
                    elif 6 < server_choose < 13:
                        server_choose -= 1
                        s[timeslot, real_ID * env.n_features:real_ID * env.n_features + env.n_features] = \
                            [work_size, vehicles_8[ID].line, servers_4[server_choose].line, cloud.line]
                        the_action = (multi_agent[real_ID].take_action(s[timeslot, real_ID * env.n_features:
                                                                             real_ID * env.n_features + env.n_features]))
                        action[timeslot, real_ID] = int(the_action)
                        if action[timeslot, real_ID] == 1:
                            n_to_server_4[server_choose] += 1
                        elif action[timeslot, real_ID] == 2:
                            n_to_cloud += 1
                    elif server_choose > 13:
                        server_choose -= 2
                        s[timeslot, real_ID * env.n_features:real_ID * env.n_features + env.n_features] = \
                            [work_size, vehicles_8[ID].line, servers_4[server_choose].line, cloud.line]
                        the_action = (multi_agent[real_ID].take_action(s[timeslot, real_ID * env.n_features:
                                                                             real_ID * env.n_features + env.n_features]))
                        action[timeslot, real_ID] = int(the_action)
                        if action[timeslot, real_ID] == 1:
                            n_to_server_4[server_choose] += 1
                        elif action[timeslot, real_ID] == 2:
                            n_to_cloud += 1
                    else:
                        s[timeslot, real_ID * env.n_features:real_ID * env.n_features + env.n_features] = \
                            [work_size, vehicles_8[ID].line, servers_4[server_choose].line, cloud.line]
                        the_action = (multi_agent[real_ID].take_action(s[timeslot, real_ID * env.n_features:
                                                                             real_ID * env.n_features + env.n_features]))
                        action[timeslot, real_ID] = int(the_action)
                        if action[timeslot, real_ID] == 1:
                            n_to_server_4[server_choose] += 1
                        elif action[timeslot, real_ID] == 2:
                            n_to_cloud += 1
                all_works[road-1, ID] = work_size
            """
            第二次循环：
                根据动作选择，计算每个任务的执行时间
                并判断所选动作是否是最小时延动作
            """
            t_work = np.zeros(env.n_actions)
            for ID in range(n_vehicle):
                road = int((ID / every_vehicle_of_road) + 1)
                real_ID = ID
                ID = int(ID % every_vehicle_of_road)
                work_size = all_works[road-1, ID]

                # 共享RSU服务器机制，获取当前最空闲的服务器编号
                servers_line_ = []
                for i in range(n_servers[0]):
                    servers_line_.append(servers_1[i].line)
                for i in range(n_servers[1]):
                    servers_line_.append(servers_2[i].line)
                for i in range(n_servers[2]):
                    servers_line_.append(servers_3[i].line)
                for i in range(n_servers[3]):
                    servers_line_.append(servers_4[i].line)
                servers_road, free_RSU = free_rsu(servers_line_)

                if road == 1:
                    server_choose = vehicles_1[ID].get_server()
                    t_work[0] = cal_time_local(vehicles_1[ID].line, work_size, vehicles_1[ID].power)

                    if n_to_server_1[server_choose] == 0:
                        n_to_server_1[server_choose] = 1
                    if servers_1[server_choose].line <= 10000:
                        t_work[1] = cal_time_RSU(servers_1[server_choose].line, work_size, servers_1[server_choose].power,
                                                 n_to_server_1[server_choose])
                    else:
                        if servers_road == 1:
                            t_work[1] = cal_time_free_RSU(servers_1[free_RSU].line, work_size, servers_1[free_RSU].power,
                                                          n_to_server_1[server_choose])
                        elif servers_road == 2:
                            t_work[1] = cal_time_free_RSU(servers_2[free_RSU].line, work_size, servers_2[free_RSU].power,
                                                          n_to_server_1[server_choose])
                        elif servers_road == 3:
                            t_work[1] = cal_time_free_RSU(servers_3[free_RSU].line, work_size, servers_3[free_RSU].power,
                                                          n_to_server_1[server_choose])
                        elif servers_road == 4:
                            t_work[1] = cal_time_free_RSU(servers_4[free_RSU].line, work_size, servers_4[free_RSU].power,
                                                          n_to_server_1[server_choose])

                    if n_to_cloud == 0:
                        n_to_cloud = 1

                    t_work[2] = cal_time_cloud(cloud.line, work_size, cloud.power, n_to_cloud)

                    # 对于实际选中的动作，进行入队处理
                    if action[timeslot, real_ID] == 0:
                        vehicles_1[ID].pop_line(work_size)
                    elif action[timeslot, real_ID] == 1:
                        if servers_1[server_choose].line <= 10000:
                            servers_1[server_choose].pop_line(work_size)
                        else:
                            if servers_road == 1:
                                servers_1[free_RSU].pop_line(work_size)
                            elif servers_road == 2:
                                servers_2[free_RSU].pop_line(work_size)
                            elif servers_road == 3:
                                servers_3[free_RSU].pop_line(work_size)
                            elif servers_road == 4:
                                servers_4[free_RSU].pop_line(work_size)
                    elif action[timeslot, real_ID] == 2:
                        cloud.pop_line(work_size)
                elif road == 2:
                    server_choose = vehicles_2[ID].get_server()
                    t_work[0] = cal_time_local(vehicles_2[ID].line, work_size, vehicles_2[ID].power)

                    if n_to_server_1[server_choose] == 0:
                        n_to_server_1[server_choose] = 1

                    if servers_1[server_choose].line <= 10000:
                        t_work[1] = cal_time_RSU(servers_1[server_choose].line, work_size, servers_1[server_choose].power,
                                                 n_to_server_1[server_choose])
                    else:
                        if servers_road == 1:
                            t_work[1] = cal_time_free_RSU(servers_1[free_RSU].line, work_size, servers_1[free_RSU].power,
                                                          n_to_server_1[server_choose])
                        elif servers_road == 2:
                            t_work[1] = cal_time_free_RSU(servers_2[free_RSU].line, work_size, servers_2[free_RSU].power,
                                                          n_to_server_1[server_choose])
                        elif servers_road == 3:
                            t_work[1] = cal_time_free_RSU(servers_3[free_RSU].line, work_size, servers_3[free_RSU].power,
                                                          n_to_server_1[server_choose])
                        elif servers_road == 4:
                            t_work[1] = cal_time_free_RSU(servers_4[free_RSU].line, work_size, servers_4[free_RSU].power,
                                                          n_to_server_1[server_choose])

                    if n_to_cloud == 0:
                        n_to_cloud = 1

                    t_work[2] = cal_time_cloud(cloud.line, work_size, cloud.power, n_to_cloud)

                    # 对于实际选中的动作，进行入队处理
                    if action[timeslot, real_ID] == 0:
                        vehicles_2[ID].pop_line(work_size)
                    elif action[timeslot, real_ID] == 1:
                        if servers_1[server_choose].line <= 10000:
                            servers_1[server_choose].pop_line(work_size)
                        else:
                            if servers_road == 1:
                                servers_1[free_RSU].pop_line(work_size)
                            elif servers_road == 2:
                                servers_2[free_RSU].pop_line(work_size)
                            elif servers_road == 3:
                                servers_3[free_RSU].pop_line(work_size)
                            elif servers_road == 4:
                                servers_4[free_RSU].pop_line(work_size)

                    elif action[timeslot, real_ID] == 2:
                        cloud.pop_line(work_size)
                elif road == 3:
                    server_choose = vehicles_3[ID].get_server()
                    t_work[0] = cal_time_local(vehicles_3[ID].line, work_size, vehicles_3[ID].power)

                    if n_to_server_2[server_choose] == 0:
                        n_to_server_2[server_choose] = 1

                    if servers_2[server_choose].line <= 10000:
                        t_work[1] = cal_time_RSU(servers_2[server_choose].line, work_size,
                                                 servers_2[server_choose].power,
                                                 n_to_server_2[server_choose])
                    else:
                        if servers_road == 1:
                            t_work[1] = cal_time_free_RSU(servers_1[free_RSU].line, work_size, servers_1[free_RSU].power,
                                                          n_to_server_2[server_choose])
                        elif servers_road == 2:
                            t_work[1] = cal_time_free_RSU(servers_2[free_RSU].line, work_size, servers_2[free_RSU].power,
                                                          n_to_server_2[server_choose])
                        elif servers_road == 3:
                            t_work[1] = cal_time_free_RSU(servers_3[free_RSU].line, work_size, servers_3[free_RSU].power,
                                                          n_to_server_2[server_choose])
                        elif servers_road == 4:
                            t_work[1] = cal_time_free_RSU(servers_4[free_RSU].line, work_size, servers_4[free_RSU].power,
                                                          n_to_server_2[server_choose])

                    if n_to_cloud == 0:
                        n_to_cloud = 1

                    t_work[2] = cal_time_cloud(cloud.line, work_size, cloud.power, n_to_cloud)
                    # 对于实际选中的动作，进行入队处理
                    if action[timeslot, real_ID] == 0:
                        vehicles_3[ID].pop_line(work_size)
                    elif action[timeslot, real_ID] == 1:
                        if servers_1[server_choose].line <= 10000:
                            servers_2[server_choose].pop_line(work_size)
                        else:
                            if servers_road == 1:
                                servers_1[free_RSU].pop_line(work_size)
                            elif servers_road == 2:
                                servers_2[free_RSU].pop_line(work_size)
                            elif servers_road == 3:
                                servers_3[free_RSU].pop_line(work_size)
                            elif servers_road == 4:
                                servers_4[free_RSU].pop_line(work_size)

                    elif action[timeslot, real_ID] == 2:
                        cloud.pop_line(work_size)
                elif road == 4:
                    server_choose = vehicles_4[ID].get_server()
                    t_work[0] = cal_time_local(vehicles_4[ID].line, work_size, vehicles_4[ID].power)

                    if n_to_server_2[server_choose] == 0:
                        n_to_server_2[server_choose] = 1

                    if servers_2[server_choose].line <= 10000:
                        t_work[1] = cal_time_RSU(servers_2[server_choose].line, work_size,
                                                 servers_2[server_choose].power,
                                                 n_to_server_2[server_choose])
                    else:
                        if servers_road == 1:
                            t_work[1] = cal_time_free_RSU(servers_1[free_RSU].line, work_size, servers_1[free_RSU].power,
                                                          n_to_server_2[server_choose])
                        elif servers_road == 2:
                            t_work[1] = cal_time_free_RSU(servers_2[free_RSU].line, work_size, servers_2[free_RSU].power,
                                                          n_to_server_2[server_choose])
                        elif servers_road == 3:
                            t_work[1] = cal_time_free_RSU(servers_3[free_RSU].line, work_size, servers_3[free_RSU].power,
                                                          n_to_server_2[server_choose])
                        elif servers_road == 4:
                            t_work[1] = cal_time_free_RSU(servers_4[free_RSU].line, work_size, servers_4[free_RSU].power,
                                                          n_to_server_2[server_choose])

                    if n_to_cloud == 0:
                        n_to_cloud = 1

                    t_work[2] = cal_time_cloud(cloud.line, work_size, cloud.power, n_to_cloud)
                    # 对于实际选中的动作，进行入队处理
                    if action[timeslot, real_ID] == 0:
                        vehicles_4[ID].pop_line(work_size)
                    elif action[timeslot, real_ID] == 1:
                        if servers_1[server_choose].line <= 10000:
                            servers_2[server_choose].pop_line(work_size)
                        else:
                            if servers_road == 1:
                                servers_1[free_RSU].pop_line(work_size)
                            elif servers_road == 2:
                                servers_2[free_RSU].pop_line(work_size)
                            elif servers_road == 3:
                                servers_3[free_RSU].pop_line(work_size)
                            elif servers_road == 4:
                                servers_4[free_RSU].pop_line(work_size)
                    elif action[timeslot, real_ID] == 2:
                        cloud.pop_line(work_size)
                elif road == 5:
                    server_choose = vehicles_5[ID].get_server()
                    t_work[0] = cal_time_local(vehicles_5[ID].line, work_size, vehicles_5[ID].power)

                    if n_to_cloud == 0:
                        n_to_cloud = 1
                    t_work[2] = cal_time_cloud(cloud.line, work_size, cloud.power, n_to_cloud)

                    if server_choose == 6:
                        if n_to_server_1[6] == 0:
                            n_to_server_1[6] = 1

                        if servers_1[6].line <= 10000:
                            t_work[1] = cal_time_RSU(servers_1[6].line, work_size, servers_1[6].power, n_to_server_1[6])
                        else:
                            if servers_road == 1:
                                t_work[1] = cal_time_free_RSU(servers_1[free_RSU].line, work_size,
                                                              servers_1[free_RSU].power,
                                                              n_to_server_1[6])
                            elif servers_road == 2:
                                t_work[1] = cal_time_free_RSU(servers_2[free_RSU].line, work_size,
                                                              servers_2[free_RSU].power,
                                                              n_to_server_1[6])
                            elif servers_road == 3:
                                t_work[1] = cal_time_free_RSU(servers_3[free_RSU].line, work_size,
                                                              servers_3[free_RSU].power,
                                                              n_to_server_1[6])
                            elif servers_road == 4:
                                t_work[1] = cal_time_free_RSU(servers_4[free_RSU].line, work_size,
                                                              servers_4[free_RSU].power,
                                                              n_to_server_1[6])

                        # 对于实际选中的动作，进行入队处理
                        if action[timeslot, real_ID] == 0:
                            vehicles_5[ID].pop_line(work_size)
                        elif action[timeslot, real_ID] == 1:
                            if servers_1[6].line <= 10000:
                                servers_1[6].pop_line(work_size)
                            else:
                                if servers_road == 1:
                                    servers_1[free_RSU].pop_line(work_size)
                                elif servers_road == 2:
                                    servers_2[free_RSU].pop_line(work_size)
                                elif servers_road == 3:
                                    servers_3[free_RSU].pop_line(work_size)
                                elif servers_road == 4:
                                    servers_4[free_RSU].pop_line(work_size)

                        elif action[timeslot, real_ID] == 2:
                            cloud.pop_line(work_size)
                    elif server_choose == 13:
                        if n_to_server_2[6] == 0:
                            n_to_server_2[6] = 1
                        if servers_2[6].line <= 10000:
                            t_work[1] = cal_time_RSU(servers_2[6].line, work_size, servers_2[6].power, n_to_server_2[6])
                        else:
                            if servers_road == 1:
                                t_work[1] = cal_time_free_RSU(servers_1[free_RSU].line, work_size,
                                                              servers_1[free_RSU].power,
                                                              n_to_server_2[6])
                            elif servers_road == 2:
                                t_work[1] = cal_time_free_RSU(servers_2[free_RSU].line, work_size,
                                                              servers_2[free_RSU].power,
                                                              n_to_server_2[6])
                            elif servers_road == 3:
                                t_work[1] = cal_time_free_RSU(servers_3[free_RSU].line, work_size,
                                                              servers_3[free_RSU].power,
                                                              n_to_server_2[6])
                            elif servers_road == 4:
                                t_work[1] = cal_time_free_RSU(servers_4[free_RSU].line, work_size,
                                                              servers_4[free_RSU].power,
                                                              n_to_server_2[6])

                        if action[timeslot, real_ID] == 0:
                            vehicles_5[ID].pop_line(work_size)
                        elif action[timeslot, real_ID] == 1:
                            if servers_2[6].line <= 10000:
                                servers_2[6].pop_line(work_size)
                            else:
                                if servers_road == 1:
                                    servers_1[free_RSU].pop_line(work_size)
                                elif servers_road == 2:
                                    servers_2[free_RSU].pop_line(work_size)
                                elif servers_road == 3:
                                    servers_3[free_RSU].pop_line(work_size)
                                elif servers_road == 4:
                                    servers_4[free_RSU].pop_line(work_size)

                        elif action[timeslot, real_ID] == 2:
                            cloud.pop_line(work_size)
                    elif 6 < server_choose < 13:
                        server_choose -= 1
                        if n_to_server_3[server_choose] == 0:
                            n_to_server_3[server_choose] = 1
                        if servers_3[server_choose].line <= 10000:
                            t_work[1] = cal_time_RSU(servers_3[server_choose].line, work_size,
                                                     servers_3[server_choose].power,
                                                     n_to_server_3[server_choose])
                        else:
                            if servers_road == 1:
                                t_work[1] = cal_time_free_RSU(servers_1[free_RSU].line, work_size,
                                                              servers_1[free_RSU].power,
                                                              n_to_server_3[server_choose])
                            elif servers_road == 2:
                                t_work[1] = cal_time_free_RSU(servers_2[free_RSU].line, work_size,
                                                              servers_2[free_RSU].power,
                                                              n_to_server_3[server_choose])
                            elif servers_road == 3:
                                t_work[1] = cal_time_free_RSU(servers_3[free_RSU].line, work_size,
                                                              servers_3[free_RSU].power,
                                                              n_to_server_3[server_choose])
                            elif servers_road == 4:
                                t_work[1] = cal_time_free_RSU(servers_4[free_RSU].line, work_size,
                                                              servers_4[free_RSU].power,
                                                              n_to_server_3[server_choose])

                        if action[timeslot, real_ID] == 0:
                            vehicles_5[ID].pop_line(work_size)
                        elif action[timeslot, real_ID] == 1:
                            if servers_3[server_choose].line <= 10000:
                                servers_3[server_choose].pop_line(work_size)
                            else:
                                if servers_road == 1:
                                    servers_1[free_RSU].pop_line(work_size)
                                elif servers_road == 2:
                                    servers_2[free_RSU].pop_line(work_size)
                                elif servers_road == 3:
                                    servers_3[free_RSU].pop_line(work_size)
                                elif servers_road == 4:
                                    servers_4[free_RSU].pop_line(work_size)

                        elif action[timeslot, real_ID] == 2:
                            cloud.pop_line(work_size)
                    elif server_choose > 13:
                        server_choose -= 2
                        if n_to_server_3[server_choose] == 0:
                            n_to_server_3[server_choose] = 1
                        if servers_3[server_choose].line <= 10000:
                            t_work[1] = cal_time_RSU(servers_3[server_choose].line, work_size,
                                                     servers_3[server_choose].power,
                                                     n_to_server_3[server_choose])
                        else:
                            if servers_road == 1:
                                t_work[1] = cal_time_free_RSU(servers_1[free_RSU].line, work_size,
                                                              servers_1[free_RSU].power,
                                                              n_to_server_3[server_choose])
                            elif servers_road == 2:
                                t_work[1] = cal_time_free_RSU(servers_2[free_RSU].line, work_size,
                                                              servers_2[free_RSU].power,
                                                              n_to_server_3[server_choose])
                            elif servers_road == 3:
                                t_work[1] = cal_time_free_RSU(servers_3[free_RSU].line, work_size,
                                                              servers_3[free_RSU].power,
                                                              n_to_server_3[server_choose])
                            elif servers_road == 4:
                                t_work[1] = cal_time_free_RSU(servers_4[free_RSU].line, work_size,
                                                              servers_4[free_RSU].power,
                                                              n_to_server_3[server_choose])
                        if action[timeslot, real_ID] == 0:
                            vehicles_5[ID].pop_line(work_size)
                        elif action[timeslot, real_ID] == 1:
                            if servers_3[server_choose].line <= 10000:
                                servers_3[server_choose].pop_line(work_size)
                            else:
                                if servers_road == 1:
                                    servers_1[free_RSU].pop_line(work_size)
                                elif servers_road == 2:
                                    servers_2[free_RSU].pop_line(work_size)
                                elif servers_road == 3:
                                    servers_3[free_RSU].pop_line(work_size)
                                elif servers_road == 4:
                                    servers_4[free_RSU].pop_line(work_size)

                        elif action[timeslot, real_ID] == 2:
                            cloud.pop_line(work_size)
                    else:
                        if n_to_server_3[server_choose] == 0:
                            n_to_server_3[server_choose] = 1
                        if servers_3[server_choose].line <= 10000:
                            t_work[1] = cal_time_RSU(servers_3[server_choose].line, work_size,
                                                     servers_3[server_choose].power,
                                                     n_to_server_3[server_choose])
                        else:
                            if servers_road == 1:
                                t_work[1] = cal_time_free_RSU(servers_1[free_RSU].line, work_size,
                                                              servers_1[free_RSU].power,
                                                              n_to_server_3[server_choose])
                            elif servers_road == 2:
                                t_work[1] = cal_time_free_RSU(servers_2[free_RSU].line, work_size,
                                                              servers_2[free_RSU].power,
                                                              n_to_server_3[server_choose])
                            elif servers_road == 3:
                                t_work[1] = cal_time_free_RSU(servers_3[free_RSU].line, work_size,
                                                              servers_3[free_RSU].power,
                                                              n_to_server_3[server_choose])
                            elif servers_road == 4:
                                t_work[1] = cal_time_free_RSU(servers_4[free_RSU].line, work_size,
                                                              servers_4[free_RSU].power,
                                                              n_to_server_3[server_choose])
                        if action[timeslot, real_ID] == 0:
                            vehicles_5[ID].pop_line(work_size)
                        elif action[timeslot, real_ID] == 1:
                            if servers_3[server_choose].line <= 10000:
                                servers_3[server_choose].pop_line(work_size)
                            else:
                                if servers_road == 1:
                                    servers_1[free_RSU].pop_line(work_size)
                                elif servers_road == 2:
                                    servers_2[free_RSU].pop_line(work_size)
                                elif servers_road == 3:
                                    servers_3[free_RSU].pop_line(work_size)
                                elif servers_road == 4:
                                    servers_4[free_RSU].pop_line(work_size)

                        elif action[timeslot, real_ID] == 2:
                            cloud.pop_line(work_size)
                elif road == 6:
                    server_choose = vehicles_6[ID].get_server()
                    t_work[0] = cal_time_local(vehicles_6[ID].line, work_size, vehicles_6[ID].power)

                    if n_to_cloud == 0:
                        n_to_cloud = 1
                    t_work[2] = cal_time_cloud(cloud.line, work_size, cloud.power, n_to_cloud)

                    if server_choose == 6:
                        if n_to_server_1[6] == 0:
                            n_to_server_1[6] = 1
                        if servers_1[6].line <= 10000:
                            t_work[1] = cal_time_RSU(servers_1[6].line, work_size, servers_1[6].power, n_to_server_1[6])
                        else:
                            if servers_road == 1:
                                t_work[1] = cal_time_free_RSU(servers_1[free_RSU].line, work_size,
                                                              servers_1[free_RSU].power,
                                                              n_to_server_1[6])
                            elif servers_road == 2:
                                t_work[1] = cal_time_free_RSU(servers_2[free_RSU].line, work_size,
                                                              servers_2[free_RSU].power,
                                                              n_to_server_1[6])
                            elif servers_road == 3:
                                t_work[1] = cal_time_free_RSU(servers_3[free_RSU].line, work_size,
                                                              servers_3[free_RSU].power,
                                                              n_to_server_1[6])
                            elif servers_road == 4:
                                t_work[1] = cal_time_free_RSU(servers_4[free_RSU].line, work_size,
                                                              servers_4[free_RSU].power,
                                                              n_to_server_1[6])

                        if action[timeslot, real_ID] == 0:
                            vehicles_6[ID].pop_line(work_size)
                        elif action[timeslot, real_ID] == 1:
                            if servers_1[6].line <= 10000:
                                servers_1[6].pop_line(work_size)
                            else:
                                if servers_road == 1:
                                    servers_1[free_RSU].pop_line(work_size)
                                elif servers_road == 2:
                                    servers_2[free_RSU].pop_line(work_size)
                                elif servers_road == 3:
                                    servers_3[free_RSU].pop_line(work_size)
                                elif servers_road == 4:
                                    servers_4[free_RSU].pop_line(work_size)

                        elif action[timeslot, real_ID] == 2:
                            cloud.pop_line(work_size)
                    elif server_choose == 13:
                        if n_to_server_2[6] == 0:
                            n_to_server_2[6] = 1
                        if servers_2[6].line <= 10000:
                            t_work[1] = cal_time_RSU(servers_2[6].line, work_size, servers_2[6].power, n_to_server_2[6])
                        else:
                            if servers_road == 1:
                                t_work[1] = cal_time_free_RSU(servers_1[free_RSU].line, work_size,
                                                              servers_1[free_RSU].power,
                                                              n_to_server_2[6])
                            elif servers_road == 2:
                                t_work[1] = cal_time_free_RSU(servers_2[free_RSU].line, work_size,
                                                              servers_2[free_RSU].power,
                                                              n_to_server_2[6])
                            elif servers_road == 3:
                                t_work[1] = cal_time_free_RSU(servers_3[free_RSU].line, work_size,
                                                              servers_3[free_RSU].power,
                                                              n_to_server_2[6])
                            elif servers_road == 4:
                                t_work[1] = cal_time_free_RSU(servers_4[free_RSU].line, work_size,
                                                              servers_4[free_RSU].power,
                                                              n_to_server_2[6])

                        if action[timeslot, real_ID] == 0:
                            vehicles_6[ID].pop_line(work_size)
                        elif action[timeslot, real_ID] == 1:
                            if servers_2[6].line <= 10000:
                                servers_2[6].pop_line(work_size)
                            else:
                                if servers_road == 1:
                                    servers_1[free_RSU].pop_line(work_size)
                                elif servers_road == 2:
                                    servers_2[free_RSU].pop_line(work_size)
                                elif servers_road == 3:
                                    servers_3[free_RSU].pop_line(work_size)
                                elif servers_road == 4:
                                    servers_4[free_RSU].pop_line(work_size)

                        elif action[timeslot, real_ID] == 2:
                            cloud.pop_line(work_size)
                    elif 6 < server_choose < 13:
                        server_choose -= 1
                        if n_to_server_3[server_choose] == 0:
                            n_to_server_3[server_choose] = 1
                        if servers_3[server_choose].line <= 10000:
                            t_work[1] = cal_time_RSU(servers_3[server_choose].line, work_size,
                                                     servers_3[server_choose].power,
                                                     n_to_server_3[server_choose])
                        else:
                            if servers_road == 1:
                                t_work[1] = cal_time_free_RSU(servers_1[free_RSU].line, work_size,
                                                              servers_1[free_RSU].power,
                                                              n_to_server_3[server_choose])
                            elif servers_road == 2:
                                t_work[1] = cal_time_free_RSU(servers_2[free_RSU].line, work_size,
                                                              servers_2[free_RSU].power,
                                                              n_to_server_3[server_choose])
                            elif servers_road == 3:
                                t_work[1] = cal_time_free_RSU(servers_3[free_RSU].line, work_size,
                                                              servers_3[free_RSU].power,
                                                              n_to_server_3[server_choose])
                            elif servers_road == 4:
                                t_work[1] = cal_time_free_RSU(servers_4[free_RSU].line, work_size,
                                                              servers_4[free_RSU].power,
                                                              n_to_server_3[server_choose])

                        if action[timeslot, real_ID] == 0:
                            vehicles_6[ID].pop_line(work_size)
                        elif action[timeslot, real_ID] == 1:
                            if servers_3[server_choose].line <= 10000:
                                servers_3[server_choose].pop_line(work_size)
                            else:
                                if servers_road == 1:
                                    servers_1[free_RSU].pop_line(work_size)
                                elif servers_road == 2:
                                    servers_2[free_RSU].pop_line(work_size)
                                elif servers_road == 3:
                                    servers_3[free_RSU].pop_line(work_size)
                                elif servers_road == 4:
                                    servers_4[free_RSU].pop_line(work_size)

                        elif action[timeslot, real_ID] == 2:
                            cloud.pop_line(work_size)
                    elif server_choose > 13:
                        server_choose -= 2
                        if n_to_server_3[server_choose] == 0:
                            n_to_server_3[server_choose] = 1
                        if servers_3[server_choose].line <= 10000:
                            t_work[1] = cal_time_RSU(servers_3[server_choose].line, work_size,
                                                     servers_3[server_choose].power,
                                                     n_to_server_3[server_choose])
                        else:
                            if servers_road == 1:
                                t_work[1] = cal_time_free_RSU(servers_1[free_RSU].line, work_size,
                                                              servers_1[free_RSU].power,
                                                              n_to_server_3[server_choose])
                            elif servers_road == 2:
                                t_work[1] = cal_time_free_RSU(servers_2[free_RSU].line, work_size,
                                                              servers_2[free_RSU].power,
                                                              n_to_server_3[server_choose])
                            elif servers_road == 3:
                                t_work[1] = cal_time_free_RSU(servers_3[free_RSU].line, work_size,
                                                              servers_3[free_RSU].power,
                                                              n_to_server_3[server_choose])
                            elif servers_road == 4:
                                t_work[1] = cal_time_free_RSU(servers_4[free_RSU].line, work_size,
                                                              servers_4[free_RSU].power,
                                                              n_to_server_3[server_choose])

                        if action[timeslot, real_ID] == 0:
                            vehicles_6[ID].pop_line(work_size)
                        elif action[timeslot, real_ID] == 1:
                            if servers_3[server_choose].line <= 10000:
                                servers_3[server_choose].pop_line(work_size)
                            else:
                                if servers_road == 1:
                                    servers_1[free_RSU].pop_line(work_size)
                                elif servers_road == 2:
                                    servers_2[free_RSU].pop_line(work_size)
                                elif servers_road == 3:
                                    servers_3[free_RSU].pop_line(work_size)
                                elif servers_road == 4:
                                    servers_4[free_RSU].pop_line(work_size)

                        elif action[timeslot, real_ID] == 2:
                            cloud.pop_line(work_size)
                    else:
                        if n_to_server_3[server_choose] == 0:
                            n_to_server_3[server_choose] = 1
                        if servers_3[server_choose].line <= 10000:
                            t_work[1] = cal_time_RSU(servers_3[server_choose].line, work_size,
                                                     servers_3[server_choose].power,
                                                     n_to_server_3[server_choose])
                        else:
                            if servers_road == 1:
                                t_work[1] = cal_time_free_RSU(servers_1[free_RSU].line, work_size,
                                                              servers_1[free_RSU].power,
                                                              n_to_server_3[server_choose])
                            elif servers_road == 2:
                                t_work[1] = cal_time_free_RSU(servers_2[free_RSU].line, work_size,
                                                              servers_2[free_RSU].power,
                                                              n_to_server_3[server_choose])
                            elif servers_road == 3:
                                t_work[1] = cal_time_free_RSU(servers_3[free_RSU].line, work_size,
                                                              servers_3[free_RSU].power,
                                                              n_to_server_3[server_choose])
                            elif servers_road == 4:
                                t_work[1] = cal_time_free_RSU(servers_4[free_RSU].line, work_size,
                                                              servers_4[free_RSU].power,
                                                              n_to_server_3[server_choose])

                        if action[timeslot, real_ID] == 0:
                            vehicles_6[ID].pop_line(work_size)
                        elif action[timeslot, real_ID] == 1:
                            if servers_3[server_choose].line <= 10000:
                                servers_3[server_choose].pop_line(work_size)
                            else:
                                if servers_road == 1:
                                    servers_1[free_RSU].pop_line(work_size)
                                elif servers_road == 2:
                                    servers_2[free_RSU].pop_line(work_size)
                                elif servers_road == 3:
                                    servers_3[free_RSU].pop_line(work_size)
                                elif servers_road == 4:
                                    servers_4[free_RSU].pop_line(work_size)

                        elif action[timeslot, real_ID] == 2:
                            cloud.pop_line(work_size)
                elif road == 7:
                    server_choose = vehicles_7[ID].get_server()
                    t_work[0] = cal_time_local(vehicles_7[ID].line, work_size, vehicles_7[ID].power)

                    if n_to_cloud == 0:
                        n_to_cloud = 1
                    t_work[2] = cal_time_cloud(cloud.line, work_size, cloud.power, n_to_cloud)

                    if server_choose == 6:
                        if n_to_server_1[13] == 0:
                            n_to_server_1[13] = 1
                        if servers_1[13].line <= 10000:
                            t_work[1] = cal_time_RSU(servers_1[13].line, work_size, servers_1[13].power, n_to_server_1[13])
                        else:
                            if servers_road == 1:
                                t_work[1] = cal_time_free_RSU(servers_1[free_RSU].line, work_size,
                                                              servers_1[free_RSU].power,
                                                              n_to_server_1[13])
                            elif servers_road == 2:
                                t_work[1] = cal_time_free_RSU(servers_2[free_RSU].line, work_size,
                                                              servers_2[free_RSU].power,
                                                              n_to_server_1[13])
                            elif servers_road == 3:
                                t_work[1] = cal_time_free_RSU(servers_3[free_RSU].line, work_size,
                                                              servers_3[free_RSU].power,
                                                              n_to_server_1[13])
                            elif servers_road == 4:
                                t_work[1] = cal_time_free_RSU(servers_4[free_RSU].line, work_size,
                                                              servers_4[free_RSU].power,
                                                              n_to_server_1[13])

                        if action[timeslot, real_ID] == 0:
                            vehicles_7[ID].pop_line(work_size)
                        elif action[timeslot, real_ID] == 1:
                            if servers_1[13].line <= 10000:
                                servers_1[13].pop_line(work_size)
                            else:
                                if servers_road == 1:
                                    servers_1[free_RSU].pop_line(work_size)
                                elif servers_road == 2:
                                    servers_2[free_RSU].pop_line(work_size)
                                elif servers_road == 3:
                                    servers_3[free_RSU].pop_line(work_size)
                                elif servers_road == 4:
                                    servers_4[free_RSU].pop_line(work_size)

                        elif action[timeslot, real_ID] == 2:
                            cloud.pop_line(work_size)
                    elif server_choose == 13:
                        if n_to_server_2[13] == 0:
                            n_to_server_2[13] = 1
                        if servers_2[13].line <= 10000:
                            t_work[1] = cal_time_RSU(servers_2[13].line, work_size, servers_2[13].power, n_to_server_2[13])
                        else:
                            if servers_road == 1:
                                t_work[1] = cal_time_free_RSU(servers_1[free_RSU].line, work_size,
                                                              servers_1[free_RSU].power,
                                                              n_to_server_2[13])
                            elif servers_road == 2:
                                t_work[1] = cal_time_free_RSU(servers_2[free_RSU].line, work_size,
                                                              servers_2[free_RSU].power,
                                                              n_to_server_2[13])
                            elif servers_road == 3:
                                t_work[1] = cal_time_free_RSU(servers_3[free_RSU].line, work_size,
                                                              servers_3[free_RSU].power,
                                                              n_to_server_2[13])
                            elif servers_road == 4:
                                t_work[1] = cal_time_free_RSU(servers_4[free_RSU].line, work_size,
                                                              servers_4[free_RSU].power,
                                                              n_to_server_2[13])

                        if action[timeslot, real_ID] == 0:
                            vehicles_7[ID].pop_line(work_size)
                        elif action[timeslot, real_ID] == 1:
                            if servers_2[13].line <= 10000:
                                servers_2[13].pop_line(work_size)
                            else:
                                if servers_road == 1:
                                    servers_1[free_RSU].pop_line(work_size)
                                elif servers_road == 2:
                                    servers_2[free_RSU].pop_line(work_size)
                                elif servers_road == 3:
                                    servers_3[free_RSU].pop_line(work_size)
                                elif servers_road == 4:
                                    servers_4[free_RSU].pop_line(work_size)

                        elif action[timeslot, real_ID] == 2:
                            cloud.pop_line(work_size)
                    elif 6 < server_choose < 13:
                        server_choose -= 1
                        if n_to_server_4[server_choose] == 0:
                            n_to_server_4[server_choose] = 1
                        if servers_4[server_choose].line <= 10000:
                            t_work[1] = cal_time_RSU(servers_4[server_choose].line, work_size,
                                                     servers_4[server_choose].power,
                                                     n_to_server_4[server_choose])
                        else:
                            if servers_road == 1:
                                t_work[1] = cal_time_free_RSU(servers_1[free_RSU].line, work_size,
                                                              servers_1[free_RSU].power,
                                                              n_to_server_4[server_choose])
                            elif servers_road == 2:
                                t_work[1] = cal_time_free_RSU(servers_2[free_RSU].line, work_size,
                                                              servers_2[free_RSU].power,
                                                              n_to_server_4[server_choose])
                            elif servers_road == 3:
                                t_work[1] = cal_time_free_RSU(servers_3[free_RSU].line, work_size,
                                                              servers_3[free_RSU].power,
                                                              n_to_server_4[server_choose])
                            elif servers_road == 4:
                                t_work[1] = cal_time_free_RSU(servers_4[free_RSU].line, work_size,
                                                              servers_4[free_RSU].power,
                                                              n_to_server_4[server_choose])

                        if action[timeslot, real_ID] == 0:
                            vehicles_7[ID].pop_line(work_size)
                        elif action[timeslot, real_ID] == 1:
                            if servers_4[server_choose].line <= 10000:
                                servers_4[server_choose].pop_line(work_size)
                            else:
                                if servers_road == 1:
                                    servers_1[free_RSU].pop_line(work_size)
                                elif servers_road == 2:
                                    servers_2[free_RSU].pop_line(work_size)
                                elif servers_road == 3:
                                    servers_3[free_RSU].pop_line(work_size)
                                elif servers_road == 4:
                                    servers_4[free_RSU].pop_line(work_size)

                        elif action[timeslot, real_ID] == 2:
                            cloud.pop_line(work_size)
                    elif server_choose > 13:
                        server_choose -= 2
                        if n_to_server_4[server_choose] == 0:
                            n_to_server_4[server_choose] = 1
                        if servers_4[server_choose].line <= 10000:
                            t_work[1] = cal_time_RSU(servers_4[server_choose].line, work_size,
                                                     servers_4[server_choose].power,
                                                     n_to_server_4[server_choose])
                        else:
                            if servers_road == 1:
                                t_work[1] = cal_time_free_RSU(servers_1[free_RSU].line, work_size,
                                                              servers_1[free_RSU].power,
                                                              n_to_server_4[server_choose])
                            elif servers_road == 2:
                                t_work[1] = cal_time_free_RSU(servers_2[free_RSU].line, work_size,
                                                              servers_2[free_RSU].power,
                                                              n_to_server_4[server_choose])
                            elif servers_road == 3:
                                t_work[1] = cal_time_free_RSU(servers_3[free_RSU].line, work_size,
                                                              servers_3[free_RSU].power,
                                                              n_to_server_4[server_choose])
                            elif servers_road == 4:
                                t_work[1] = cal_time_free_RSU(servers_4[free_RSU].line, work_size,
                                                              servers_4[free_RSU].power,
                                                              n_to_server_4[server_choose])

                        if action[timeslot, real_ID] == 0:
                            vehicles_7[ID].pop_line(work_size)
                        elif action[timeslot, real_ID] == 1:
                            if servers_4[server_choose].line <= 10000:
                                servers_4[server_choose].pop_line(work_size)
                            else:
                                if servers_road == 1:
                                    servers_1[free_RSU].pop_line(work_size)
                                elif servers_road == 2:
                                    servers_2[free_RSU].pop_line(work_size)
                                elif servers_road == 3:
                                    servers_3[free_RSU].pop_line(work_size)
                                elif servers_road == 4:
                                    servers_4[free_RSU].pop_line(work_size)

                        elif action[timeslot, real_ID] == 2:
                            cloud.pop_line(work_size)
                    else:
                        if n_to_server_4[server_choose] == 0:
                            n_to_server_4[server_choose] = 1
                        if servers_4[server_choose].line <= 10000:
                            t_work[1] = cal_time_RSU(servers_4[server_choose].line, work_size,
                                                     servers_4[server_choose].power,
                                                     n_to_server_4[server_choose])
                        else:
                            if servers_road == 1:
                                t_work[1] = cal_time_free_RSU(servers_1[free_RSU].line, work_size,
                                                              servers_1[free_RSU].power,
                                                              n_to_server_4[server_choose])
                            elif servers_road == 2:
                                t_work[1] = cal_time_free_RSU(servers_2[free_RSU].line, work_size,
                                                              servers_2[free_RSU].power,
                                                              n_to_server_4[server_choose])
                            elif servers_road == 3:
                                t_work[1] = cal_time_free_RSU(servers_3[free_RSU].line, work_size,
                                                              servers_3[free_RSU].power,
                                                              n_to_server_4[server_choose])
                            elif servers_road == 4:
                                t_work[1] = cal_time_free_RSU(servers_4[free_RSU].line, work_size,
                                                              servers_4[free_RSU].power,
                                                              n_to_server_4[server_choose])

                        if action[timeslot, real_ID] == 0:
                            vehicles_7[ID].pop_line(work_size)
                        elif action[timeslot, real_ID] == 1:
                            if servers_4[server_choose].line <= 10000:
                                servers_4[server_choose].pop_line(work_size)
                            else:
                                if servers_road == 1:
                                    servers_1[free_RSU].pop_line(work_size)
                                elif servers_road == 2:
                                    servers_2[free_RSU].pop_line(work_size)
                                elif servers_road == 3:
                                    servers_3[free_RSU].pop_line(work_size)
                                elif servers_road == 4:
                                    servers_4[free_RSU].pop_line(work_size)

                        elif action[timeslot, real_ID] == 2:
                            cloud.pop_line(work_size)
                elif road == 8:
                    server_choose = vehicles_8[ID].get_server()
                    t_work[0] = cal_time_local(vehicles_8[ID].line, work_size, vehicles_8[ID].power)

                    if n_to_cloud == 0:
                        n_to_cloud = 1
                    t_work[2] = cal_time_cloud(cloud.line, work_size, cloud.power, n_to_cloud)

                    if server_choose == 6:
                        if n_to_server_1[13] == 0:
                            n_to_server_1[13] = 1
                        if servers_1[13].line <= 10000:
                            t_work[1] = cal_time_RSU(servers_1[13].line, work_size, servers_1[13].power, n_to_server_1[13])
                        else:
                            if servers_road == 1:
                                t_work[1] = cal_time_free_RSU(servers_1[free_RSU].line, work_size,
                                                              servers_1[free_RSU].power,
                                                              n_to_server_1[13])
                            elif servers_road == 2:
                                t_work[1] = cal_time_free_RSU(servers_2[free_RSU].line, work_size,
                                                              servers_2[free_RSU].power,
                                                              n_to_server_1[13])
                            elif servers_road == 3:
                                t_work[1] = cal_time_free_RSU(servers_3[free_RSU].line, work_size,
                                                              servers_3[free_RSU].power,
                                                              n_to_server_1[13])
                            elif servers_road == 4:
                                t_work[1] = cal_time_free_RSU(servers_4[free_RSU].line, work_size,
                                                              servers_4[free_RSU].power,
                                                              n_to_server_1[13])

                        if action[timeslot, real_ID] == 0:
                            vehicles_8[ID].pop_line(work_size)
                        elif action[timeslot, real_ID] == 1:
                            if servers_1[13].line <= 10000:
                                servers_1[13].pop_line(work_size)
                            else:
                                if servers_road == 1:
                                    servers_1[free_RSU].pop_line(work_size)
                                elif servers_road == 2:
                                    servers_2[free_RSU].pop_line(work_size)
                                elif servers_road == 3:
                                    servers_3[free_RSU].pop_line(work_size)
                                elif servers_road == 4:
                                    servers_4[free_RSU].pop_line(work_size)
                        elif action[timeslot, real_ID] == 2:
                            cloud.pop_line(work_size)
                    elif server_choose == 13:
                        if n_to_server_2[13] == 0:
                            n_to_server_2[13] = 1
                        if servers_2[13].line <= 10000:
                            t_work[1] = cal_time_RSU(servers_2[13].line, work_size, servers_2[13].power, n_to_server_2[13])
                        else:
                            if servers_road == 1:
                                t_work[1] = cal_time_free_RSU(servers_1[free_RSU].line, work_size,
                                                              servers_1[free_RSU].power,
                                                              n_to_server_2[13])
                            elif servers_road == 2:
                                t_work[1] = cal_time_free_RSU(servers_2[free_RSU].line, work_size,
                                                              servers_2[free_RSU].power,
                                                              n_to_server_2[13])
                            elif servers_road == 3:
                                t_work[1] = cal_time_free_RSU(servers_3[free_RSU].line, work_size,
                                                              servers_3[free_RSU].power,
                                                              n_to_server_2[13])
                            elif servers_road == 4:
                                t_work[1] = cal_time_free_RSU(servers_4[free_RSU].line, work_size,
                                                              servers_4[free_RSU].power,
                                                              n_to_server_2[13])

                        if action[timeslot, real_ID] == 0:
                            vehicles_8[ID].pop_line(work_size)
                        elif action[timeslot, real_ID] == 1:
                            if servers_2[13].line <= 10000:
                                servers_2[13].pop_line(work_size)
                            else:
                                if servers_road == 1:
                                    servers_1[free_RSU].pop_line(work_size)
                                elif servers_road == 2:
                                    servers_2[free_RSU].pop_line(work_size)
                                elif servers_road == 3:
                                    servers_3[free_RSU].pop_line(work_size)
                                elif servers_road == 4:
                                    servers_4[free_RSU].pop_line(work_size)

                        elif action[timeslot, real_ID] == 2:
                            cloud.pop_line(work_size)
                    elif 6 < server_choose < 13:
                        server_choose -= 1
                        if n_to_server_4[server_choose] == 0:
                            n_to_server_4[server_choose] = 1
                        if servers_4[server_choose].line <= 10000:
                            t_work[1] = cal_time_RSU(servers_4[server_choose].line, work_size,
                                                     servers_4[server_choose].power,
                                                     n_to_server_4[server_choose])
                        else:
                            if servers_road == 1:
                                t_work[1] = cal_time_free_RSU(servers_1[free_RSU].line, work_size,
                                                              servers_1[free_RSU].power,
                                                              n_to_server_4[server_choose])
                            elif servers_road == 2:
                                t_work[1] = cal_time_free_RSU(servers_2[free_RSU].line, work_size,
                                                              servers_2[free_RSU].power,
                                                              n_to_server_4[server_choose])
                            elif servers_road == 3:
                                t_work[1] = cal_time_free_RSU(servers_3[free_RSU].line, work_size,
                                                              servers_3[free_RSU].power,
                                                              n_to_server_4[server_choose])
                            elif servers_road == 4:
                                t_work[1] = cal_time_free_RSU(servers_4[free_RSU].line, work_size,
                                                              servers_4[free_RSU].power,
                                                              n_to_server_4[server_choose])

                        if action[timeslot, real_ID] == 0:
                            vehicles_8[ID].pop_line(work_size)
                        elif action[timeslot, real_ID] == 1:
                            if servers_4[server_choose].line <= 10000:
                                servers_4[server_choose].pop_line(work_size)
                            else:
                                if servers_road == 1:
                                    servers_1[free_RSU].pop_line(work_size)
                                elif servers_road == 2:
                                    servers_2[free_RSU].pop_line(work_size)
                                elif servers_road == 3:
                                    servers_3[free_RSU].pop_line(work_size)
                                elif servers_road == 4:
                                    servers_4[free_RSU].pop_line(work_size)

                        elif action[timeslot, real_ID] == 2:
                            cloud.pop_line(work_size)
                    elif server_choose > 13:
                        server_choose -= 2
                        if n_to_server_4[server_choose] == 0:
                            n_to_server_4[server_choose] = 1
                        if servers_4[server_choose].line <= 10000:
                            t_work[1] = cal_time_RSU(servers_4[server_choose].line, work_size,
                                                     servers_4[server_choose].power,
                                                     n_to_server_4[server_choose])
                        else:
                            if servers_road == 1:
                                t_work[1] = cal_time_free_RSU(servers_1[free_RSU].line, work_size,
                                                              servers_1[free_RSU].power,
                                                              n_to_server_4[server_choose])
                            elif servers_road == 2:
                                t_work[1] = cal_time_free_RSU(servers_2[free_RSU].line, work_size,
                                                              servers_2[free_RSU].power,
                                                              n_to_server_4[server_choose])
                            elif servers_road == 3:
                                t_work[1] = cal_time_free_RSU(servers_3[free_RSU].line, work_size,
                                                              servers_3[free_RSU].power,
                                                              n_to_server_4[server_choose])
                            elif servers_road == 4:
                                t_work[1] = cal_time_free_RSU(servers_4[free_RSU].line, work_size,
                                                              servers_4[free_RSU].power,
                                                              n_to_server_4[server_choose])

                        if action[timeslot, real_ID] == 0:
                            vehicles_8[ID].pop_line(work_size)
                        elif action[timeslot, real_ID] == 1:
                            if servers_4[server_choose].line <= 10000:
                                servers_4[server_choose].pop_line(work_size)
                            else:
                                if servers_road == 1:
                                    servers_1[free_RSU].pop_line(work_size)
                                elif servers_road == 2:
                                    servers_2[free_RSU].pop_line(work_size)
                                elif servers_road == 3:
                                    servers_3[free_RSU].pop_line(work_size)
                                elif servers_road == 4:
                                    servers_4[free_RSU].pop_line(work_size)

                        elif action[timeslot, real_ID] == 2:
                            cloud.pop_line(work_size)
                    else:
                        if n_to_server_4[server_choose] == 0:
                            n_to_server_4[server_choose] = 1
                        if servers_4[server_choose].line <= 10000:
                            t_work[1] = cal_time_RSU(servers_4[server_choose].line, work_size,
                                                     servers_4[server_choose].power,
                                                     n_to_server_4[server_choose])
                        else:
                            if servers_road == 1:
                                t_work[1] = cal_time_free_RSU(servers_1[free_RSU].line, work_size,
                                                              servers_1[free_RSU].power,
                                                              n_to_server_4[server_choose])
                            elif servers_road == 2:
                                t_work[1] = cal_time_free_RSU(servers_2[free_RSU].line, work_size,
                                                              servers_2[free_RSU].power,
                                                              n_to_server_4[server_choose])
                            elif servers_road == 3:
                                t_work[1] = cal_time_free_RSU(servers_3[free_RSU].line, work_size,
                                                              servers_3[free_RSU].power,
                                                              n_to_server_4[server_choose])
                            elif servers_road == 4:
                                t_work[1] = cal_time_free_RSU(servers_4[free_RSU].line, work_size,
                                                              servers_4[free_RSU].power,
                                                              n_to_server_4[server_choose])

                        if action[timeslot, real_ID] == 0:
                            vehicles_8[ID].pop_line(work_size)
                        elif action[timeslot, real_ID] == 1:
                            if servers_4[server_choose].line <= 10000:
                                servers_4[server_choose].pop_line(work_size)
                            else:
                                if servers_road == 1:
                                    servers_1[free_RSU].pop_line(work_size)
                                elif servers_road == 2:
                                    servers_2[free_RSU].pop_line(work_size)
                                elif servers_road == 3:
                                    servers_3[free_RSU].pop_line(work_size)
                                elif servers_road == 4:
                                    servers_4[free_RSU].pop_line(work_size)

                        elif action[timeslot, real_ID] == 2:
                            cloud.pop_line(work_size)

                min_time_action = np.argmin(t_work)
                min_time = np.min(t_work)

                the_action = int(action[timeslot, real_ID])

                if min_time_action == the_action:
                    r[timeslot, real_ID] = 1
                else:
                    r[timeslot, real_ID] = -1

                t_timeslots[timeslot] += t_work[the_action]

            # 处理任务
            servers_line = []
            vehicle_line = []
            # print('云服务器的队列长度为：', cloud.line)
            cloud.handle_works()
            for i in range(n_servers[0]):
                servers_line.append(servers_1[i].line)
                servers_1[i].handle_works()
                # servers_line.append(servers_1[i].line)
            for i in range(n_servers[1]):
                servers_line.append(servers_1[i].line)
                servers_2[i].handle_works()
                # servers_line.append(servers_2[i].line)
            for i in range(n_servers[2]):
                servers_line.append(servers_1[i].line)
                servers_3[i].handle_works()
                # servers_line.append(servers_3[i].line)
            for i in range(n_servers[3]):
                servers_line.append(servers_1[i].line)
                servers_4[i].handle_works()
                # servers_line.append(servers_4[i].line)
            for ID in range(n_road_vehicle):
                vehicle_line.append(vehicles_1[ID].line)
                vehicle_line.append(vehicles_2[ID].line)
                vehicle_line.append(vehicles_3[ID].line)
                vehicle_line.append(vehicles_4[ID].line)
                vehicle_line.append(vehicles_5[ID].line)
                vehicle_line.append(vehicles_6[ID].line)
                vehicle_line.append(vehicles_7[ID].line)
                vehicle_line.append(vehicles_8[ID].line)

                vehicles_1[ID].forward()
                vehicles_2[ID].forward()
                vehicles_3[ID].forward()
                vehicles_4[ID].forward()
                vehicles_5[ID].forward()
                vehicles_6[ID].forward()
                vehicles_7[ID].forward()
                vehicles_8[ID].forward()

            # print('RSU服务器的队列长度为：', servers_line)
            # print('车辆的队列长度为：', vehicle_line)

        # print(r)
        # print(action)
        print(sum(t_timeslots))
        t_epochs[test, epoch] = sum(t_timeslots)

        """
            一个epoch中所有时间片结束之后
            需要的数据都已经按时间片统计好
            按智能体为主体，进行数据的封装和学习
        """
        # 获取s_，将经验存入记忆库
        for ID in range(n_vehicle):
            for timeslot in range(n_timeslot):
                if timeslot < n_timeslot - 1:
                    s_[timeslot, ID * env.n_features:ID * env.n_features + env.n_features] \
                        = s[timeslot + 1, ID * env.n_features:ID * env.n_features + env.n_features]
                else:
                    s_[timeslot, ID * env.n_features:ID * env.n_features + env.n_features] = \
                        s[timeslot, ID * env.n_features:ID * env.n_features + env.n_features]

                # 将数据添加到经验池
                # buffer.add(state, action, reward, next_state, done)
                multi_replay_buffer[ID].add(
                    s[timeslot, ID * env.n_features:ID * env.n_features + env.n_features],
                    action[timeslot, ID],
                    r[timeslot, ID],
                    s_[timeslot, ID * env.n_features:ID * env.n_features + env.n_features],
                    done=False)

        # 训练
        for _ in range(n_train):
            for ID in range(n_vehicle):
                s, a, r, ns, d = multi_replay_buffer[ID].sample(batch_size)  # 每次取出batch组数据
                # 构造数据集
                transition_dict = {'states': s,
                                   'actions': a,
                                   'rewards': r,
                                   'next_states': ns,
                                   'dones': d}
                # 模型训练
                multi_agent[ID].update(transition_dict)

print(t_epochs)
# 将数组转换为 DataFrame
df = pd.DataFrame(t_epochs)
# 将 DataFrame 保存为 Excel 文件
df.to_excel('output.xlsx')
