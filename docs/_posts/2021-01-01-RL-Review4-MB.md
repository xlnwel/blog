---
title: "RL Reviews 4 — Model-Based RL"
excerpt: "The forth post in a series of reviews of algorithms in reinforcement learning"
categories:
  - Reinforcement Learning
tags:
  - RL Review
---

## Introduction

In this post, we will review model-free reinforcement learning algorithms. 

**Caveat**: For the sake of simplicity, this series is made to be as concise as possible and only serve as the purpose of review. Links to the post with detailed discussion are provided for the algorithms that have been studied in this blog. 

## <a name="dir"></a>Table of Contents

- [Given Model](#given)
  - [RTDP](#rtdp)
  - [iLQR](#lqr)
  - [MCTS](#mcts)
  - [ExIt](#exit)
  - [AlphaZero](#alphazero)
- [Learned Model](#learned)
  - [GPS with iLQR](#gps)
  - [MBMF](#mbmf)
  - [TDM](#tdm)
  - [MB-MPO](#mbmpo)
  - [PlaNet](#planet)
  - [Dreamer](#dreamer)
  - [DreamerV2](#dreamerv2)
  - [MuZero](#muzero)
- [Adaptive Model](#adaptive)

## <a name="given"></a>Learn Value Function

### <a name="rtdp"></a>[Real-Time Dynamic Programming]({{ site.baseurl }}{% post_url 2018-11-14-planning %})

Real-Time Dynamic Programming(RTDP) is a dynamic-programming algorithm that extends LRTA* to stochastic environments. It uses the Bellman optimality equation(as shown below) to update the value function at the current state and states from there on if time is available. 

$$
\begin{align}
V(s)=\max_a \sum_{s',r'\sim p(s',r'|s,a)}\left(r+\gamma V(s')\right)
\end{align}
$$

RTDP is guaranteed to find an optimal policy on the relevant states for *stochastic optimal path problems*, which usually stated in terms of *cost minimization*—in which all rewards are negative—instead of reward maximization. As LRTA*, $$V(s)$$ here also serves as a heuristic function; as a result, $$V(s)$$ must be optimistically initialized(e.g., to zero).

[Elevator back to directory](#dir)

### <a name="lqr"></a>[iterative Linear Quadratic Regulator]({{ site.baseurl }}{% post_url 2018-12-14-iLQR %})

Linear Quadratic Regulator(LQR) solves a linear dynamics system with a quadratic reward function by constructing value functions and auxiliary matrices from time step $$T$$ to the beginning.

The same process can be applied to the case where the dynamics model is a linear-Gaussian distribution with a constant covariance.

For a non-linear system, **i**terative LQR(iLQR) solve them by taking the Taylor expansion to approximate a linear system at some trajectory. Then it solves the linear-quadratic approximation via LQR and runs a simulation to get a new trajectory, from which it repeats the above process.

[Elevator back to directory](#dir)

### <a name="mcts"></a>[Monte-Carlo Tree Search]({{ site.baseurl }}{% post_url 2018-11-14-planning %})

Monte-Carlo Tree Search is a search algorithm, notably for their application in board games. It repeats the following steps:

1. Perform the tree policy(UCT) in the tree to find a leaf node
2. Expand the leaf node by adding some child nodes to the tree
3. Run simulation from the newly added children according to the rollout policy(e.g., uniform random policy)
4. Backpropagate the simulation results up to the tree root

[Elevator back to directory](#dir)

### <a name="exit"></a>Expert Iteration

<figure>
  <img src="{{ '/images/model-based/ExIt.png' | absolute_url }}" alt="" width="1000">
  <figcaption></figcaption>
  <style>
    figure figcaption {
    text-align: center;
    }
  </style>
</figure>
[**Ex**pert **It**eration(ExIt)](https://arxiv.org/abs/1705.08439) is an algorithm for board games. It uses MCTS as the expert policy to supervise an apprentice policy. The optimization objective is to minimize the cross-entropy between the expert and apprentice($$\pi$$) policies

$$
\begin{align}
\mathcal L=-\sum_{a}{n(s,a)\over n(s)}\log\pi(a|s)
\end{align}
$$

The MCTS in ExIt is refined in two aspects

1. To bias MCTS towards moves the apprentice believes to be stronger, UCT in MCTS is augmented as
   
$$
   UCT_{ExIt}(s,a)=UCT(s,a)+w_a{\pi(a|s)\over n(s,a)+1}
   $$

2. MCTS uses apprentice's value-function($$V(s)$$) to shorten the horizon. The value function shares most of the layers with the policy network and is trained by minimizing the cross-entropy between the sampled outcome $$z$$ and $$V(s)$$.
   
$$
   \mathcal L_V=-z\log V(s)-(1-z)\log(1-V(s))
   $$


[Elevator back to directory](#dir)

### <a name="alphazero"></a>[AlphaZero]({{ site.baseurl }}{% post_url 2021-02-21-AlphaZero %})

AlphaZero runs MCTS and selects action based on the root visit count—either proportionally or greedily. Instead of running simulations from leaf nodes, it uses a neural network to predict the state value and move probabilities from a leaf node. The neural network is a deep ResNet with two head, one for the policy and the other for the value function. The policy is trained to minimize the cross entropy to the MCTS policy, while the value function is trained to predict the result without any discounting.

To encourage exploration, a Dirichlet noise is applied to the root of the search tree.

AlphaZero chooses MCTS rather than alpha-beta search because the deep ResNet may introduce a large worst-case generalization error. When combined with alpha-beta search, the errors are propagated directly to the root of the subtree via the minmax operation. By contrast, MCTS averages over the position evaluations, which may cancel out the approximation error introduced by the network.

[Elevator back to directory](#dir)

## <a name="learned"></a>Learned Model

### <a name="gps"></a>[Guided Policy Search with iLQR]({{ site.baseurl }}{% post_url 2018-12-21-GPS %})

Guided Policy Search with iLQR produces a time-varying linear-Gaussian policy by repeating the following steps

1. Run the current policy to sample a trajectory
2. Fit the local model(a linear Gaussian) using linear regression
3. Improve the policy using dual gradient descent

where step 3 solves the constrained optimization problem

$$
\begin{align}
\max \sum_{t=1}^T\mathbb E_{p(s_t,a_t)}\left[r(s_t,a_t)\right]\\\
s.t.\ \sum_{t=1}^TD_{KL}(p(a_t|s_t)\Vert p_{old}(a_t|s_t))\le \epsilon
\end{align}
$$

by alternating between the following two steps

1. Find the optimal $$p$$ that maximizes the Lagrangian using LQR(or some KL regularized method)
2. Perform gradient descent on the Lagrangian multiplier.

[Elevator back to directory](#dir)

### <a name="mbmf"></a>[Model-Based Model-Free]({{ site.baseurl }}{% post_url 2018-12-07-MBMF %})

Model-Based Model-Free(MBMF) first learns an MPC controller, then uses the MPC controller to initialize a model-free algorithm through imitation learning with DAgger. Finally, it runs the model-free algorithm to further improve performance.

When learning the model, MBMF has a network predict the difference between the next and the current state: $$f(s,a)=\hat s'-s$$ and optimizes it with loss:

$$
\begin{align}
\mathcal L(\theta)=\mathbb E_{}\left[{1\over H}\sum_{h=1}^H{1\over 2}\Vert s_{t+h}-\hat s_{t+h}\Vert^2\right]\\\
\mathrm{where\quad}\hat s_{t+h}=\begin{cases}s_t& h=0\\\
\hat s_{t+h-1}+f(\hat s_{t+h-1}, a_{t+h-1})&h>0\end{cases}
\end{align}
$$


[Elevator back to directory](#dir)

### <a name="tdm"></a>[Temporal Difference Models]({{ site.baseurl }}{% post_url 2019-04-27-TDM %})

Temporal Difference Models(TDMs), designed for goal-directed tasks, trains a family of goal-conditioned value functions with model-free learning and uses them for model-based control.

The $$Q$$-function in TDMs is defined as

$$
\begin{align}
Q(s_t,a_t,s_g,\tau)=-|f(s_t,a_t,s_g,\tau)-s_g|
\end{align}
$$

where $$f$$ is a parametric model trained to predict the state that will be reached in $$\tau$$ steps by a policy attempting to reach $$s_g$$. $$\vert \cdot\vert $$ is the absolute difference in each dimension or L1 norm depending on whether the TDM is vector-valued or scalar.

The $$Q$$-function is trained using some off-policy algorithm with target 

$$
\begin{align}
y_m=-|s_{t+1}-s_{g,m}|\mathbf 1[\tau_m=0]-|f(s_{t+1},a^*,s_{g,m},\tau_m-1)-s_{g,m}|\mathbf 1[\tau_m\ne 0]\\\
where\ a^*=\underset{a}{\arg\max}Q(s_{t+1},a,s_{g,m},\tau_m-1)
\end{align}
$$

where $$s_{g,m}$$ is sampled from future states along the actual trajectory in the buffer, and $$\tau_m$$ from $$[0, \tau_\max]$$. 

TDMs rely on MPC to make decisions. Ideally, it can do so in a single step

$$
\begin{align}
a_t=\underset{a_t,a_{t+T},s_{t+T}}{\arg\max} r_c(f(s_t,a_t,s_{t+T},T-1),a_{t+T})\tag{5}\label{eq:5}
\end{align}
$$

[Elevator back to directory](#dir)

### <a name="mbmpo"></a>[Model-Based Meta-Policy Optimization]({{ site.baseurl }}{% post_url 2019-07-14-MB-MPO %})

Model-Based Meta-Policy Optimization learns an ensemble of dynamics models and runs MAML to learn a meta-policy that can adapt for all learned models. 

Learning an adaptive policy imposes a regularizing effect on the policy learning, and mitigate the over-conservative issue of a policy that is robust across models.

MB-MPO samples trajectories from the real environment with the adapted policies. This increase the diversity of the training data, which promotes robustness of the dynamics models

[Elevator back to directory](#dir)

### <a name="planet"></a>[PlaNet]({{ site.baseurl }}{% post_url 2020-02-14-PlaNet %})

<figure>
  <img src="{{ '/images/model-based/PlaNet-figure2.png' | absolute_url }}" alt="" width="1000">
  <figcaption></figcaption>
  <style>
    figure figcaption {
    text-align: center;
    }
  </style>
</figure>

PlaNet introduces the recurrent state-space model(RSSM), which uses a deterministic state module(a GRU) to maintain the sequential information and a stochastic model to serve as a belief state. Models are trained with reconstruction loss as shown in Figure 2c.

PlaNet make decisions based on the cross entropy method.

[Elevator back to directory](#dir)

### <a name="dreamer"></a>[Dreamer]({{ site.baseurl }}{% post_url 2020-02-27-Dreamer %})

Dreamer augments PlaNet by training an additional AC model from the imagined trajectories. As the imagined trajectories are on-policy, the actor is trained to maximize $$TD(\lambda)$$ and the value function is trained to predict $$TD(\lambda)$$.

[Elevator back to directory](#dir)

### <a name="dreamerv2"></a>[DreamerV2]({{ site.baseurl }}{% post_url 2020-03-01-DreamerV2 %})

DreamerV2 modifies Dreamer to achieve promising results on Atari games. The changes include

- **Categorical latents.** Using categorical latent states using straight-through gradients in the world model instead of Gaussian latents with reparameterized gradients.
- **Mixed actor gradients.** Combining Reinforce and dynamics backpropagation gradients for learning the actor instead of dynamics backpropagation only.

- **Policy entropy.** Regularizing the policy entropy for exploration both in imagination and during data collection, instead of using external action noise during data collection.

- **KL balancing.** Separately scaling the prior cross entropy and the posterior entropy in the KL loss to encourage learning an accurate temporal prior, instead of using free nats.

- **Model size.** Increasing the number of units or feature maps per layer of all model components, resulting in a change from 13M parameters to 22M parameters.

- **Layer norm.** Using layer normalization in the GRU that is used as part of the RSSM latent transition model, instead of no normalization.

[Elevator back to directory](#dir)

### <a name="muzero"></a>[MuZero]({{ site.baseurl }}{% post_url 2021-02-27-MuZero %})

MuZero, the successor of AlphaZero, dispenses with the simulator and trains a representation model and a dynamic model to learn the hidden dynamics. Both the representation and the dynamics models are a deep ResNet. The representation model maps the a history of observations to a latent space, from which the dynamics model predict the immediate reward and the next state. Mathematically, we have

$$
\begin{align}
\text{representation model:}&&s^0=&h_\theta(o_1,\dots,o_t)\\\
\text{dynamics model:}&&r^k,s^k=&g_\theta(s^{k-1},a^k)\\\
\text{AC model:}&&p^k,v^k=&f_\theta(s^k)
\end{align}
$$

Notice that MuZero assumes deterministic transition dynamics.

During training, MuZero unrolls the network for $$K=5$$ hypothetical steps. To maintain a roughly similar magnitude of gradient across different unroll steps, we 1) scale the loss of each head by $$1\over K$$ and 2) scale the gradient at the start of the dynamics function by $$1/2$$. Moreover, the hidden state are scaled to the same range as the action: $$s_{scale}={s-\min(s)\over\max(s)-\min(s)}$$.

For Atari, the value function is updated with $$n=10$$-step bootstrapped target. 

[Elevator back to directory](#dir)