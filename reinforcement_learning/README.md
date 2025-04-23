# Reinforcement Learning

# `rl_max_simple_policy_gradient.ipynb`

## Enviroment
Create a `N*N` maze. The agent can only observe it's surroundings (local 3 * 3 grid).

Based on the observations, the agent is expected to take a move, go up/down/left/right.

Four boundaries are walled. The agent always enters at (1,1). The agent aims to exit at (N-2,N-2) (Success).

## Rewards

Instant reward of each step is not clear util the agent reaches the maze exit at (N-2,N-2).

The expected reward at starting position `x_0` will be `V(x_0) = sum prob(d|x_0_grid_3x3) * V(x_1)` where moving from `x_0` toward direction `d` will get to `x_1` (in this maze example, this is deterministic)

We can define `V((N-2,N-2))` = 1 -- reward is 1 if the agent reaches the maze exit.

## Policy Gradient

We are not interested in absolute reward, but interested in the policy `prob(d|x_grid_3x3)`, ie. which direction to go given the local `3*3` grid.

We can have a neural net taking `x_grid_3x3` as input, outputing the probability for all four directions `prob(up),prob(down),prob(left),prob(right)`.

Best policy maximizes `V(x_0)`. Take the derivative over the neural net weight:

``dV(x_0) / dw = sum_{d_0} dprob(d_0) / dw *  d V(x_1) / dw``

Expand this we got all pathes `path = {x_0,d_0,x_1,d_1...}`,

``dV(x_0) / dw = sum_{path} prod{i} dprob(d_i | x_i_grid_3x3) / dw``

If we *define* `V((N-2,N-2))` = 1 -- reward is 1 if the agent reaches the maze exit. Otherwise, the reward is -1 after `max_step` reached.

Then the above derivative over network weight has the same loss function over a single sample path (stochastic gradient descent over a mini-batch):

`log-loss = sum_{d_i} log prob(d_i | x_i_grid_3x3)` if the path reached dest (N-2,N-2)

or 

`log-loss = sum_{d_i} log prob(d_i | x_i_grid_3x3)` if `max_step` is reached.

## On-policy learning

We can learn while we simulate. The agent will initially randomly traverse the maze. The learning won't start until it accidently reached the destination for the first time. The network weight gets updated -- the agent learns that it should try to go bottom/right, and then it learns how to go over a blocker.





