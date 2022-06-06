import gym
import minerl
import numpy as np

env = gym.make('MineRLNavigate-v0')
# print('OS: ', len(env.observation_space))
# print(env.observation_space)
# print(env.action_space)
# print('AS: ', len(env.action_space))
Q = []
# print(Q)
# print([len(env.observation_space),len(env.action_space)])
alpha = .628
gma = .9
epis = 5000

def getIndex(state, action):
    index = 0
    for s, a, _ in Q:
        # print('S: ', str(s))
        # print('STATE: ', str(state))
        if str(s) == str(state) and str(a) == str(action):
            return index
        else:
            index += 1

def getActionFromQTable(state):
    bestAction = None
    bestReward = 0
    for s, a, r in Q:
        # print('s ', s)
        # print('state ', state)
        if str(s) == str(state):
            if r > bestReward:
                bestAction = a
    return bestAction

def getMaxReward(state):
    bestReward = 0
    for s, _, r in Q:
        if str(s) == str(state):
            if r > bestReward:
                bestReward = r
    return bestReward


def updateQTable(state, action, newState, reward):
    index = getIndex(state, action)
    if index != None:
        print('INDEX: ', index)
    currReward = 0
    if index == None:
        newReward = currReward + alpha*(reward + gma*getMaxReward(newState) - currReward)
        Q.append((state, action, newReward))
    else:
        _, _, currReward = Q[index]
        newReward = currReward + alpha*(reward + gma*getMaxReward(newState) - currReward)
        Q[index] = (state, action, newReward)

for i in range(epis):
    state = env.reset()
    # print('state: ', state)
    done = False
    # j = 0
    while not done:
        env.render()
        # j += 1
        # print('Q size: ', len(Q))
        # Choose action from Q table
        action = getActionFromQTable(state)
        if action == None:
            action = env.action_space.sample()
            # action['camera'] = [0, 0.03*state["compass"]["angle"]]
            # action['back'] = 0
            # action['forward'] = 1
            # action['jump'] = 1
            # action['attack'] = 1
            # action['sprint'] = 0
        # Get new state and reward from environment
        newState, reward, done, _ = env.step(action)
        # Update Q-Table with new information
        updateQTable(state, action, newState, reward)
        state = newState
        if done == True:
            break
    env.render()
