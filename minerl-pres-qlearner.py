import gym
import minerl
import numpy as np
# from minerl.data import BufferedBatchIter

# data = minerl.data.make('MineRLTreechop-v0')
env = gym.make('MineRLTreechop-v0')
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
    # while j < 99:
    while not done:
        env.render()
        # j += 1
        # print('Q size: ', len(Q))
        # Choose action from Q table
        action = getActionFromQTable(state)
        if action == None:
            action = env.action_space.noop()
            # print(action)
            # print(env.action_space)
            willJump = 0
            jumpProb = np.random.randint(0, 11)
            if jumpProb == 1:
                willJump = 1
            willForward = 0
            forwardProb = np.random.randint(0, 3)
            if forwardProb == 1:
                willForward = 1
            willBack = 0
            backProb = np.random.randint(0, 11)
            if backProb == 1:
                willBack = 1
            willLeft = 0
            leftProb = np.random.randint(0, 11)
            if leftProb == 1:
                willLeft = 1
            willRight = 0
            rightProb = np.random.randint(0, 11)
            if rightProb == 1:
                willRight = 1
            action['camera'] = [0, 1*np.random.randint(-5, 4)]
            action['back'] = willBack
            action['left'] = willLeft
            action['right'] = willRight
            action['forward'] = willForward
            action['jump'] = willJump
            action['attack'] = 1
            action['sprint'] = 0
            action['sneak'] = 0
        # Get new state and reward from environment
        newState, reward, done, _ = env.step(action)
        if reward != 0:
            print('REWARD: ', reward)
        # Update Q-Table with new information
        updateQTable(state, action, newState, reward)
        state = newState
        if done == True:
            break
    env.render()
