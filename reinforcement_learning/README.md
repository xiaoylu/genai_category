# Reinforcement Learning

## Maze Example 

### Environment
Create a `N*N` maze. The agent can only observe its surroundings (local 3 × 3 grid). $x_0^{\text{grid}_{3\times3}}$ denotes the observations of the local 3X3 grid centered at $x_0$.

Based on the observations, the agent is expected to take a move: go up/down/left/right, the direction is denoted by $d$.

Four boundaries are walled. The agent always enters at (1,1). The agent aims to exit at (N-2,N-2) (Success).

In this 9*9 maze, we start from *, and the goal is reach #.

```
[1, 1, 1, 1, 1, 1, 1, 1, 1]
[1, *, 0, 1, 1, 0, 0, 0, 1]
[1, 1, 0, 0, 0, 0, 0, 0, 1]
[1, 0, 0, 0, 0, 1, 0, 0, 1]
[1, 0, 0, 0, 0, 0, 0, 1, 1]
[1, 0, 0, 1, 0, 0, 0, 0, 1]
[1, 1, 1, 0, 0, 0, 0, 0, 1]
[1, 1, 0, 0, 0, 0, 0, #, 1]
[1, 1, 1, 1, 1, 1, 1, 1, 1]
```

### Rewards

Instant reward of each step is not clear until the agent reaches the maze exit at (N-2,N-2).

> The credit assignment problem occurs when an agent receives a reward (or punishment) that is delayed from the actions that actually caused it. In your maze example, the agent only gets a reward (+1) when it reaches the goal, but the reward doesn't tell you which specific actions along the path were good or bad.

The expected reward at starting position $x_0$ will be:

$$V(x_0) = \sum_d P(d|x_0^{\text{grid}_{3\times3}}) \cdot V(x_1)$$

where $P(d|x_0^{\text{grid}_{3\times3}})$ is the probability moving from $x_0$ toward direction $d$, (and this will get to $x_1$). 


This is a recursive definition of the value function for each cell in the maze. To given it an initial state, we define $V((N-2,N-2)) = 1$, namely the value of exit is 1.

### Policy Gradient

We are not interested in absolute reward, but interested in the policy $P(d|x^{\text{grid}_{3\times3}})$, i.e., which direction to go, given the local $3\times3$ grid.

We will have a neural net taking $x^{\text{grid}_{3\times3}}$ as input, outputting the probability for all four directions: $P(\text{up})$, $P(\text{down})$, $P(\text{left})$, $P(\text{right})$. In other word, $P()$ is a function over neural network weights $w$.

Best policy maximizes $V(x_0)$. Take the derivative over the neural net weight:

$$\frac{dV(x_0)}{dw} = \sum_{d_0} \frac{dP(d_0)}{dw} \cdot \frac{dV(x_1)}{dw} $$

Expand this by all the possible paths of length $M$, the derivation is a sum over the product along each concrete $\text{path} = \{x_0,d_0,x_1,d_1,\ldots,x_M\}$:

$$\frac{dV(x_0)}{dw} = \sum_{\text{path}} \prod_i \frac{dP(d_i | x_i^{\text{grid}_{3\times3}})}{dw} \cdot V^{*}(x_M) $$

If we *define* $V^{*}(x_M) = 1$ if this path leads to the exit (N-2,N-2), namely the reward is 1 if the agent reaches the maze exit. 

Otherwise, the reward is $V^{*}(x_M) = -1$ after the max step $M$ is reached.

To maximize $V(x_0)$ over neural network weight $w$, we want to follow the opposite direction of the $\frac{dV(x_0)}{dw}$.

Imagine if some path reached destination $(N-2,N-2)$, minimizing this log-loss below follows the derivation as maximizing $V(x_0)$,

$$\text{log-loss} = - \sum_{d_i} \log P(d_i | x_i^{\text{grid}_{3\times3}})$$

so does when max step ${M}$ is reached, minimizing the following log-loss needs to follow the derivation as maximizing $V(x_0)$,

$$\text{log-loss} = \sum_{d_i} \log P(d_i | x_i^{\text{grid}_{3\times3}})$$

### On-policy Learning

We can learn while we simulate. The agent will traverse the maze as the neutral network instructs (draw from the categorial distribution). 

Then, once $M$ step is reach, depending on whether you reached the destination, the network weight gets updated according to the log-loss.

Check out juypter notebook [`rl_maze_simple_policy_gradient.ipynb`](./rl_maze_simple_policy_gradient.ipynb) for this implementation. You can see the number of steps needed for the agent to reach the exit will decrease over time as the $P()$ gets improved.

### PPO


