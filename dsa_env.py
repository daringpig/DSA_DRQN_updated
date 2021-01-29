import numpy as np

class DsaCliqueEnv(object):
    def __init__(self, num_user, num_channel, r_fail, r_succeed, r_idle, r_fairness=None):
        self.num_user = num_user               # 用户数
        self.num_channel = num_channel         # 通道数

        # reward
        self.r_fail = r_fail                   # 失败reward
        self.r_succeed = r_succeed             # 成功reward
        self.r_idle = r_idle                   # 空闲reward

        # fairness-aware reward
        # If a user has occupied a channel for x consecutive time slots
        # reward of success will be penalized by a factor (sigmoid function)
        # When r_fail = 0 and r_succeed = 1, reward of success will drop to 0.5 at x = f_thre
        # x will be cleared if 1. switch to another channel 2. idle 3. conflict

        # 防止一个用户占用一个通道太长时间，引入r_fairness机制
        self.f_thre = None
        if r_fairness is not None and r_fairness > 0:
            self.delta = 1 + np.exp(-1)
            self.f_thre = r_fairness

        # 每个用户i有两个存储历史信息的标志值
        # history[i,0] 表示的是选择的通道数的编号
        # history[i,1] 表示的是连续几次选择该通道数了
        self.history = np.zeros((num_user, 2), dtype=int)

        # space
        # 0 表示没有选择任何通道进行传输
        self.n_action = num_channel + 1                # 动作空间
        self.n_observation = num_channel + 2           # 观测空间

        # timestamp
        self.t = 0                                     # 计算步数

    # 当一个用户i占用一个通道的次数x越大时，回报值越小
    # 比如，当x=f_thre，也即x=fairness时，回报值只有之前的一半
    def r_succeed_fair(self, x):
        if self.f_thre is not None:
            tmp = (1 / (1 + np.exp(x / self.f_thre - 1)) - 0.5) * self.delta + 0.5
            return (self.r_succeed - self.r_fail) * tmp + self.r_fail
        else:
            return self.r_succeed

    def reset(self):
        self.t = 0
        # 初始时，或重置后，所有的user的历史通道访问信息为空
        self.history = np.zeros((self.num_user, 2), dtype=int)
        # 观测值，一个user对应的观测值是一个向量，表示这个user选择的通道
        obs = np.zeros((self.num_user, self.n_observation), dtype=float)
        # 初始时，或重置后，所有的user都没有选择任何通道进行传输
        obs[:, 0] = 1
        return obs

    # 输入：action   输出：obs，r，in_use
    # 执行一步action，获取观测值obs，回报r，通道占用情况in_use
    def step(self, action):

        # for experiments
        collision = 0

        self.t += 1                                                                  # 为此步骤计数
        in_use = np.zeros(self.num_channel, dtype=int)                               # 此步骤执行完后，每个通道的空闲状态
        r = np.zeros(self.num_user)                                                  # 每个用户一个回报值
        obs = np.zeros((self.num_user, self.n_observation), dtype=float)             # 每个用户一个观测值

        # 执行动作，获取obs和in_use
        for i in range(self.num_user):
            obs[i, action[i]] = 1                                                    # 表示第i个用户占用了第action[i]个通道
            if action[i] > 0:                                                        # 除了action[i]=0的情况，即未占用通道，其他情况下，channel的占用状态被in_use记录下来
                in_use[action[i] - 1] += 1

        # 执行动作，获取reward
        for i in range(self.num_user):
            # 选择发送数据
            if action[i] > 0:
                # 发生冲突     # conflict
                if in_use[action[i] - 1] > 1:
                    r[i] = self.r_fail
                    self.history[i, 0] = 0
                    # for experiments
                    collision += 1

                # 未发生冲突    # succeed
                else:
                    # 用户i上一个选择的通道和本次选择的一样
                    if self.history[i, 0] == action[i]:
                        r[i] = self.r_succeed_fair(self.history[i, 1])
                        self.history[i, 1] += 1

                    # 用户i上一个选择的通道和本次选择的不一样
                    else:
                        r[i] = self.r_succeed
                        self.history[i, 0] = action[i]
                        self.history[i, 1] = 1

                    # 观测值obs的最后一位表示：执行action后，用户i成功地发送了数据
                    obs[i, -1] = 1

            # 选择不发送数据
            else:
                r[i] = self.r_idle
                self.history[i, 0] = 0

        return obs, r, False, in_use, collision
