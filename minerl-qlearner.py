import gym
import minerl
import numpy as np

env = gym.make('MineRLTreechop-v0')
Q = np.zeros([len(env.observation_space),len(env.action_space)])
eta = .628
gma = .9
epis = 5000
rev_list = []

for i in range(epis):
    s = env.reset()
    print('s: ', s)
    rAll = 0
    d = False
    j = 0
    while j < 99:
        env.render()
        j+=1
        print('Q: ', Q[s,:])
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
        s1,r,d,_ = env.step(a)
        Q[s,a] = Q[s,a] + eta*(r + gma*np.max(Q[s1,:]) - Q[s,a])
        rAll += r
        s = s1
        if d == True:
            break
    rev_list.append(rAll)
    env.render()

print("Reward Sum on all episodes " + str(sum(rev_list)/epis))
print("Final Values Q-Table")
print(Q)
