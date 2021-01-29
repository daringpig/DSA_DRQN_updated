import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from agent import Agent
import os
import pickle


def train_agent(env, device, exp_name, agents, num_it, num_ep, max_ts, target_update_freq, gamma, lstm_hidden_size,
                eps_start, eps_end, eps_end_it, beta_start, beta_end):
    """
    env:                environment
    device:             cpu or gpu
    agents:             list of agents, specified by env.num_user

    num_it:             num of iterations
    num_ep:             num of episode in each iteration
    max_ts:             num of time slots in each episode

    target_update_freq: number of iterations token to update target network
    gamma:              dicount factor
    lstm_hidden_size:   the size of hidden state in LSTM layer

    eps_start:          epsilon at the beginning of training
    eps_end:            epsilon at the end of training
    eps_end_it:         number of iterations token to reach ep_end in linearly-annealed epsilon-greedy policy
    """

    # 写日志，管理日志
    log_dir = './' + exp_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = open(log_dir + 'log.txt', 'a+')
    log_file.write('# of users: {}\n'.format(str(env.num_user)))
    log_file.write('# of channels: {}\n'.format(str(env.num_channel)))
    log_file.write('R_fail: {}\n'.format(str(env.r_fail)))
    log_file.write('R_idle: {}\n'.format(str(env.r_idle)))
    log_file.write('R_succeed: {}\n'.format(str(env.r_succeed)))

    # 准备训练数据，初始时刻，训练数据全为空~
    batch_size = num_ep * max_ts
    s_batch = torch.empty((env.num_user, batch_size, env.n_observation), dtype=torch.float).to(device=device)
    s2_batch = torch.empty((env.num_user, batch_size, env.n_observation), dtype=torch.float).to(device=device)
    a_batch = torch.empty((env.num_user, batch_size), dtype=torch.int64).to(device=device)
    r_batch = torch.empty((env.num_user, batch_size), dtype=torch.float).to(device=device)
    h0_batch = torch.empty((env.num_user, batch_size, lstm_hidden_size), dtype=torch.float).to(device=device)
    h1_batch = torch.empty((env.num_user, batch_size, lstm_hidden_size), dtype=torch.float).to(device=device)
    h20_batch = torch.empty((env.num_user, batch_size, lstm_hidden_size), dtype=torch.float).to(device=device)
    h21_batch = torch.empty((env.num_user, batch_size, lstm_hidden_size), dtype=torch.float).to(device=device)
    a = np.zeros(env.num_user, dtype=int)

    # 迭代次数，也即训练的次数
    for it in tqdm(range(1, num_it + 1)):
        # 刚开始eps比较大，后面逐渐变小，而且从最大到最小在一定的步数(eps_end_it)内
        # 在1000步以内，eps从1.0逐步减小到0.01，后续一直保持0.01
        # 在1000步以内，beta从1.0逐步增加至20，后续一直持续20
        epsilon = eps_start - (eps_start - eps_end) * (it - 1) / eps_end_it if it <= eps_end_it else eps_end
        beta = beta_start + (beta_end - beta_start) * (it - 1) / eps_end_it if it <= eps_end_it else beta_end

        # sampling from environment
        cnt = 0
        avg_r = 0
        avg_utils = 0

        # for experiments
        all_r = 0   # 一个episode中的所有reward
        all_c = 0   # 一个episode中的所有的collision
        all_r_lst = []
        all_c_lst = []

        # 遍历每个episode
        for ep in range(num_ep):

            # 初始的观测值，大小为（num_user, n_observation）
            s = env.reset()

            # 初始化网络参数
            h0 = torch.normal(mean=0, std=0.01, size=(env.num_user, lstm_hidden_size)).to(device=device)
            h1 = torch.normal(mean=0, std=0.01, size=(env.num_user, lstm_hidden_size)).to(device=device)
            h20 = h0.clone()
            h21 = h1.clone()

            # for experiment
            all_r = 0
            all_c = 0
            r_lst = np.zeros([max_ts])
            c_lst = np.zeros([max_ts])

            for t in range(0, max_ts):
                for j in range(env.num_user):
                    # 根据上一步的观测状态，选择一个可以执行的动作
                    # 一个用户对应一个agent，每个time slot的训练，都要所有用户都要选择需执行的动作
                    # s[j], h0[j], h1[j] 都是一维矢量
                    a[j], (h20[j], h21[j]) = agents[j].action(s[j], h0[j], h1[j], epsilon, beta)

                # for test
                #if(t==20):
                #    print("Shape of actions: ", a.shape)

                s2, r, done, channel_status, c = env.step(a)

                # for test
                print("\nObservations for time %d: ----------" % t)
                for u_i in range(s2.shape[0]):
                    for obs_i in range(s2.shape[1]):
                        print(s2[u_i, obs_i], end=' ')
                    print("")


                # collect training samples in batch
                s_batch[:, cnt, :] = torch.from_numpy(s)
                s2_batch[:, cnt, :] = torch.from_numpy(s2)
                a_batch[:, cnt] = torch.from_numpy(a)
                r_batch[:, cnt] = torch.from_numpy(r)
                h0_batch[:, cnt, :] = h0
                h1_batch[:, cnt, :] = h1
                h20_batch[:, cnt, :] = h20
                h21_batch[:, cnt, :] = h21

                # for experiment
                all_r += r.sum()
                r_lst[t] = all_r
                all_c += c
                c_lst[t] = all_c

                avg_r += r.sum() / r.size
                avg_utils += np.sum(channel_status == 1) / channel_status.size
                cnt += 1
                s = s2
                h0 = h20.clone()
                h1 = h21.clone()

            all_r_lst.append(r_lst)
            all_c_lst.append(c_lst)

        # training
        for j in range(env.num_user):
            agents[j].train(s_batch[j], h0_batch[j], h1_batch[j], a_batch[j], r_batch[j], s2_batch[j], h20_batch[j],
                            h21_batch[j], gamma)

        if it % target_update_freq == 0:
            for j in range(env.num_user):
                agents[j].update_target()

        # print reward
        # 每训练100次，写日志记录average reward和channel utilization
        if it % 100 == 0:
            log_file.write('Iteration {}: avg reward is {:.4f}, channel utilization is {:.4f}\n'.format(it, avg_r / cnt,
                                                                                                      avg_utils / cnt))
            log_file.flush()

            # for experiment
            tmp_fr = '_' + str(it) + '_reward.pkl'
            with open(tmp_fr, "wb") as fn:
                pickle.dump(np.array(all_r_lst), fn)

            tmp_fc = '_' + str(it) + '_collision.pkl'
            with open(tmp_fc, "wb") as fn:
                pickle.dump(np.array(all_c_lst), fn)


            # for test
            print("---------------------100---------------------")

    # 管理日志
    log_file.close()

    for i in range(env.num_user):
        model = agents[i].get_model()
        torch.save(model.state_dict(), log_dir + 'agent_' + str(i) + '.model')


def eval_agent(env, device, exp_name, model_files, num_ep, max_ts, eps_end, lstm_hidden_size, beta):

    log_dir = './' + exp_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = open(log_dir + 'eval_log.txt', 'w+')

    # Load trained networks and build agents
    agents = []
    for i in range(len(model_files)):
        a = Agent(env=env, device=device, lstm_hidden_size=lstm_hidden_size)
        a.load_model_from_state_dict(torch.load(model_files[i]))
        agents.append(a)
    
    epsilon = eps_end

    # Evaluation
    for ep in range(num_ep):
        s = env.reset()
        h0 = torch.normal(mean=0, std=0.01, size=(env.num_user, lstm_hidden_size)).to(device=device)
        h1 = torch.normal(mean=0, std=0.01, size=(env.num_user, lstm_hidden_size)).to(device=device)
        h20 = h0.clone()
        h21 = h1.clone()
        a = np.zeros(env.num_user, dtype=int)
        user_r = np.zeros(env.num_user, dtype=float)
        user_s = np.zeros(env.num_user, dtype=float)
        avg_r = 0.0
        avg_utils = 0.0

        for t in range(max_ts):
            for j in range(env.num_user):
                a[j], (h20[j], h21[j]) = agents[j].action(s[j], h0[j], h1[j], epsilon, beta)
                if a[j] > 0:
                    user_s[j] += 1
            s2, r, done, channel_status, _ = env.step(a)
            user_r += r

            avg_r += r.sum() / r.size
            avg_utils += np.sum(channel_status == 1) / channel_status.size
            s = s2
            h0 = h20.clone()
            h1 = h21.clone()

        # Calculate the avg reward and sending rate
        user_r /= max_ts
        user_s /= max_ts
        avg_r /= max_ts
        avg_utils /= max_ts
        log_file.write('Episode {}: Eval results:\n'.format(str(ep)))
        log_file.write('Avg reward: {:.3f} \nAvg channel util: {:.3f}\n'.format(
            avg_r, avg_utils))
        for i in range(len(user_r)):
            log_file.write('User {}: avg reward {:.3f}; sending rate {:.3f}\n'.format(i, user_r[i], user_s[i]))

    log_file.close()


def draw_episode(env, device, exp_name, model_files, max_ts, eps_end, lstm_hidden_size, beta):
    # Load trained networks and build agents
    agents = []
    for i in range(len(model_files)):
        a = Agent(env=env, device=device, lstm_hidden_size=lstm_hidden_size)
        a.load_model_from_state_dict(torch.load(model_files[i]))
        agents.append(a)

    epsilon = eps_end

    color = ['white', 'orange', 'yellow', 'red', 'lightgreen']
    fig = plt.figure()
    ax = fig.add_subplot(111)

    s = env.reset()
    h0 = torch.normal(mean=0, std=0.01, size=(env.num_user, lstm_hidden_size)).to(device=device)
    h1 = torch.normal(mean=0, std=0.01, size=(env.num_user, lstm_hidden_size)).to(device=device)
    h20 = h0.clone()
    h21 = h1.clone()
    a = np.zeros(env.num_user, dtype=int)

    for t in range(max_ts):
        for j in range(env.num_user):
            a[j], (h20[j], h21[j]) = agents[j].action(s[j], h0[j], h1[j], epsilon, beta)
            ax.add_patch(Rectangle((t, j), 1, 1, angle=0.0, color=color[a[j]]))
        s2, r, done, channel_status, _ = env.step(a)
        s = s2
        h0 = h20.clone()
        h1 = h21.clone()

    plt.xlim([0, max_ts])
    plt.ylim([0, env.num_user])
    plt.show()
    plt.savefig('./' + exp_name + '/' + 'episode.png')