Reinforcement Learning
======================

Reinforcement learning is used when the objective is not a
differentiable function of model parameters (otherwise it is probably
more efficient to use supervised learning).  Some examples:

- Atari: we do not know the correct joystick action at each step, just
  whether we won points at each time step or not.  The function that
  goes from actions to points is not known or if known (e.g. we have
  the game emulator) it is not differentiable. If you had an oracle
  with the optimal actions you would just use supervised learning. If
  you could write the total reward as a differentiable function of
  model parameters (e.g. sequence classification RNNs) you would just
  use supervised learning.

Nevertheless we should point out the similarities to supervised
learning, because the independently developed terminology may hide the
parallels.

- Reward = negative loss = the thing we try to maximize.
- Action = prediction = output of the model at each time step.

Notation and terminology:

- Observation: $O_t$
- Reward: $R_t$
- Action: $A_t$
- Return: $G_t = R_{t+1} + \gamma R_{t+2} + \ldots = \sum_{k=0}^\infty \gamma^k R_{t+k+1}$
- Policy: $\pi(a|s)=P(A_t=a|S_t=s)$.
- State value function: $v_\pi(s)=E_\pi[G_t|S_t=s]$ prediction of discounted future reward.
- Action value function: $q_\pi(s,a)=E_\pi[G_t|S_t=s,A_t=a]$.
- History: $H_t = O_1, R_1, A_1, O_2, \ldots, A_{t-1}, O_t, R_t$
- State: $S_t = f(H_t)$ (agent state, which may or may not equal env state)
- Distinguish environment state $S_t^e$ from agent state $S_t^a$.
- Belief state: $b(h)=(P[S_t=s_1|H_t=h],\ldots,P[S_t=s_n|H_t=h])$.
- MDP: environment state fully observable: $O_t = S_t^a = S_t^e$.
- Bandit: MDP with one state.
- POMDP: partially observable markov decision process.
- POMDP states may be a function of history (e.g. RNN), or beliefs (probability distributions) over possible environment states.
- Model: prediction of environment behavior, i.e. next state, next reward.
- Depending on the algorithm policy, value function, model can be represented explicitly.
- Learning vs planning: if we have a model of the environment, optimizing behavior is called planning.
- Prediction vs control: prediction evaluates the future given a policy, control finds the   best policy.
- Markov process: is a tuple $\langle S,P\rangle$ where $S$ is a finite set of states and $$P_{ss'}$ is a state transition probability matrix. One can sample episodes from an MP, find the stationary distribution etc.
- Markov reward process: is a tuple $\langle S,P,R,\gamma\rangle$ where $R_s = E[R_{t+1}|S_t=s]$ is a reward function and $\gamma\in[0,1]$ is a discount factor. We can sample returns and compute expected return from each state (value function).
- Markov decision process: is a tuple $\langle S,A,P,R,\gamma\rangle$ where $A$ is a finite set of actions and $P^a_{ss'}$ and $R^a_s$ are also indexed with action $a$.
- Given a policy: $\pi(a|s)=P(A_t=a|S_t=s)$ for an MDP, the state sequence is an MP and the state-reward sequence is an MRP.
- There is always a deterministic optimal policy for any MDP.
- Partially observable Markov decision process: is a tuple $\langle S,A,O,P,R,Z,\gamma\rangle$ where $O$ is a finite set of observations and $Z^a_{s'o}=P[O_{t+1}=o|S_{t+1}=s',A_t=a]$ is an observation function. The complete history $H_t$ and the belief state $b(H_t)$ both satisfy the Markov property and can be used to reduce POMDP to MDP?
- Bellman expectation equation: find the value function for a given policy. $v = R + \gamma P v$ can solve for the value function for small problems. For large problems: DP?, MC, TD.
- Bellman optimality equation: find the optimal value function and policy. $v_*(s)=\max_a q_*(s,a)$, $q_*(s,a)=R_s^a+\gamma\sum_{s'\in S} P^a_{ss'} v_*(s')$. For large problems: Value iteration, policy iteration, Q-learning, SARSA.

If you have defined your problem in terms of the concepts above and
hope that RL will magically solve it, you are probably mistaken.
Almost every nontrivial problem can be posed in above terms:
e.g. numerical optimization of arbitrary functions without
derivatives, which cannot be solved efficiently in the general case.

We will focus on large problems modeled with approximations rather
than exact solutions of toy cases.

- Lec 3: Given exact model, predict value (iterative policy evaluation (using BEE)), find optimal policy (policy iteration = policy evaluation + greedy policy improvement, and value iteration (using BOE)). These are DP approaches.
- Lec 4: Without model, predict value function (MC, TD, TD(λ)).
- Lec 5: Without model, find optimal policy (MC, TD, SARSA, SARSA(λ), off-policy: MC, TD, Q-learning).
- Dimensions: full(dp) vs sample(td) vs MC backups. BEE vs BOE. V vs Q.  Est. val and improve policy or iterate value fn directly.


References
----------
-   <http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html>
-   <https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PL7-jPKtc4r78-wCZcQn5IqyuWhBZ8fOxT>
-   <http://videolectures.net/rldm2015_silver_reinforcement_learning/?q=david%20silver>
-   <https://webdocs.cs.ualberta.ca/~sutton/book/the-book.html>
-   <https://sites.ualberta.ca/~szepesva/RLBook.html>
-   <http://banditalgs.com/print/>
-   <http://karpathy.github.io/2016/05/31/rl/>
-   <http://cs229.stanford.edu/notes/cs229-notes12.pdf>
-   <http://cs.stanford.edu/people/karpathy/reinforcejs/index.html>
-   <https://www.udacity.com/course/machine-learning-reinforcement-learning>--ud820
-   <http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html>
-   <http://people.csail.mit.edu/regina/my_papers/TG15.pdf>
-   In <http://karpathy.github.io/2015/05/21/rnn-effectiveness>: For
    more about REINFORCE and more generally Reinforcement Learning and
    policy gradient methods (which REINFORCE is a special case of) David
    Silver's class, or one of Pieter Abbeel's classes. This is very much
    ongoing work but these hard attention models have been explored, for
    example, in Inferring Algorithmic Patterns with Stack-Augmented
    Recurrent Nets, Reinforcement Learning Neural Turing Machines, and
    Show Attend and Tell.
-   In <http://www.deeplearningbook.org/contents/ml.html>: Please see
    Sutton and Barto (1998) or Bertsekasand Tsitsiklis (1996) for
    information about reinforcement learning, and Mnih et al.(2013) for
    the deep learning approach to reinforcement learning.

