import gym
import minerl
from minerl.data import BufferedBatchIter
data = minerl.data.make('MineRLTreechop-v0')
iterator = BufferedBatchIter(data)
for current_state, action, reward, next_state, done \
    in iterator.buffered_batch_iter(batch_size=1, num_epochs=1):

        # Print the POV @ the first step of the sequence
        print(current_state['pov'][0])

        # Print the final reward pf the sequence!
        print(reward[-1])

        # Check if final (next_state) is terminal.
        print(done[-1])
