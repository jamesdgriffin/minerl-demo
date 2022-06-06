import gym
import minerl
import numpy as np

env = gym.make('FrozenLake8x8-v1')
Q = np.zeros([env.observation_space.n,env.action_space.n])
eta = .628
gma = .9
epis = 1
rev_list = []

for i in range(epis):
    # Reset environment
    s = env.reset()
    print('S: ', s)
    d = False
    # The Q-Table learning algorithm
    while d != True:
        env.render()
        # Choose action from Q table
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
        #Get new state & reward from environment
        s1,r,d,_ = env.step(a)
        #Update Q-Table with new knowledge
        Q[s,a] = Q[s,a] + eta*(r + gma*np.max(Q[s1,:]) - Q[s,a])
        s = s1
    # Code will stop at d == True, and render one state before it
