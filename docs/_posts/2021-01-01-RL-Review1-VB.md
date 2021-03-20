---
title: "RL Reviews 1 — Value-Based RL"
excerpt: "The first post in a series of reviews of algorithms in reinforcement learning"
categories:
  - Reinforcement Learning
tags:
  - RL Review
---

## Introduction

In this post, we will review value-based reinforcement learning algorithms. 

**Caveat**: For the sake of simplicity, this series is made to be as concise as possible and only serve as the purpose of review. Links to the post with detailed discussion are provided for the algorithms that have been studied in this blog. 

## <a name="dir"></a>Table of Contents

- [Deep Q-Learning](#dql)
    - [DQN](#dqn)
    - [Distributional Q-Learning](#dist)
        - [c51](#c51)
        - [QR-DQN](#qrdqn)
        - [IQN](#iqn)
        - [FQF](#fqf)
    - [Rainbow](#rainbow)
        - [DDQN](#ddqn)
        - [Dueling DQN](#duel)
        - [Noisy Networks](#noisy)
        - [PER](#per)
    - [PopArt](#popart)
    - [TBO](#tbo)
    - [Retrace](#retrace)
    - [R2D2](#r2d2)
    - [Agent57](#agent57)
    - [Decisions in DQN](#decisions_dqn)
- [Deterministic Policy Gradient](#dpg)
    - [DDPG](#ddpg)
    - [TD3](#td3)
    - [D4PG](#d4pg)

## <a name="dql"></a>Deep Q-Learning

### <a name="dqn"></a>[Deep Q-Network]({{ site.baseurl }}{% post_url 2018-10-01-DQN %})

Deep Q-Network(DQN) adopts three auxiliary techniques to stablize training with deep neural networks:

1. Target network
2. Experience Replay
3. Huber loss and gradient clipping

[Elevator back to directory](#dir)

### <a name="dist"></a>Distributional Q-Learning

#### <a name="c51"></a>[c51]({{ site.baseurl }}{% post_url 2018-10-21-c51 %})

C51 models a value distribution $$Z(s,a)$$ whose expectation is the action-value, i.e., $$Q(s,a)=\mathbb E[Z(s,a)]$$. Specifically, the value distribution is modeled by a discrete distribution parameterized by $$N\in\mathbb N$$ and $$V_{\min},V_{\max}\in \mathbb R$$, and whose support is the set of atoms

$$
\begin{align}
\left\{z_i=V_{\min}+i\Delta z:0\le i<N\right\}\\\
\Delta z:={V_{\max}-V_{\min}\over N-1}
\end{align}
$$

The corresponding atom probabilities are given by a parameterized model $$\theta=S\times A\rightarrow\mathbb R^N$$

$$
\begin{align}
Z_\theta(s,a)=z_i\\\
\mathrm{w.p.}\quad p_i(s,a):=\mathrm{softmax}(\theta_i(s,a))={\exp({\theta_i(s,a)})\over\sum_j\exp(\theta_j(s,a))}
\end{align}
$$

The target value distribution is defined as the projected Bellman update

$$
\begin{align}
\left(\Phi \mathcal{\hat T}Z_\theta(s, a)\right)_i=\sum_{j=0}^{N-1}\left[1-{\left|\left[\mathcal{\hat T}z_j\right]_{V_{\min}}^{V_{\max}}-z_i\right|\over\Delta z} \right]_0^1p_j(s', \pi(s'))
\end{align}
$$

and we optimize the value distribution by minimizing the KL divergence

$$
\begin{align}
D_{KL}\left(\Phi\mathcal{\hat T}Z_{\tilde \theta}(s, a)\Vert Z_\theta(s,a)\right)
\end{align}
$$


[Elevator back to directory](#dir)

#### <a name="qrdqn"></a>[Quantile Regression Deep Q Network]({{ site.baseurl }}{% post_url 2019-03-27-IQN %})

Quantile Regression Deep Q Network(QR-DQN) optimizes the quantile functions at points onto which the projection of the target distribution has the minimal Wasserstein distance. For $$N$$ equally spaced intervals $$\{[\tau_i,\tau_{i+1}]_{i=0}^{N-1}\}$$, it is easy to show that these points are the midpoints

$$
\begin{align}
\{\tau_i={\tau_i+\tau_{i+1}\over 2};0\le i< N\}\\\
\end{align}
$$

The corresponding quantile functions $$\theta$$ optimized by the quantile Huber loss

$$
\begin{align}
\mathcal L(\theta)&=\sum_{i=0}^{N-1}{1\over N}\sum_{j=0}^{N-1}\rho_{\tau_i}^\kappa(\delta_{ij}(s,a))\\\
where\quad
\rho_{\tau_i}^\kappa(u)&=\mathcal L_\kappa(u)|\tau_i-\mathbf 1_{u<0}|\\\
\quad \mathcal L_\kappa(u)&=\begin{cases}{1\over 2}u^2,&\mathrm{if\ }|u|\le\kappa\\\
\kappa(|u|-{1\over 2}\kappa),&\mathrm{otherwise}\end{cases}\\\
\delta_{ij}(s,a)&=r(s,a)+\gamma\theta_j^-(s',a^*)-\theta_i(s,a)\\\
a^*&=\underset{a'}{\arg\max}Q(s',a')\\\
Q(s,a)&=\mathbb E[\theta(s,a)]={1\over N}\sum_{i=0}^{N-1}\theta_i(s,a)\\\
\end{align}
$$


[Elevator back to directory](#dir)

#### <a name="iqn"></a>[Implicit Quantile Networks]({{ site.baseurl }}{% post_url 2019-03-27-IQN %})

Implicit Quantile Networks learns the whole quantile functions, and takes an additional input to query the quantile value at specified locations.

<figure>
  <img src="{{ '/images/distributional/iqn.png' | absolute_url }}" alt="" width="1000">
  <figcaption></figcaption>
  <style>
    figure figcaption {
    text-align: center;
    }
  </style>
</figure>

The loss is similar to that of QR-DQN, but the policy is a little different

$$
\begin{align}
\pi_\beta(s) = \underset{a\in\mathcal A}{\arg\max}{1\over K}\sum_{k=1}^K\theta_{\tilde\tau_k}(s,a),\quad\tilde\tau_k\sim\beta(\tau),\ \tau\sim U([0,1])
\end{align}
$$

where $$\beta:[0,1]\rightarrow[0,1]$$ is known as a distortion risk measure, by choosing which the agent establishes a risk-sensitive behavior.

𝛿

[Elevator back to directory](#dir)

#### <a name="fqf"></a>[Fully Parameterized Quantile Function]({{ site.baseurl }}{% post_url 2019-04-01-FQF %})

FQF introduces a fraction proposal network for proposing probabilities $$\{\tau_i\}_{i=0}^{N-1}$$. The fraction proposal network is trained jointly with the quantile function to minimize the 1-Wasserstein distance between the target and the current distributions. This gives the following gradient

$$
\begin{align}
\forall i\in(0,N),\quad{\partial W_1\over\partial\tau_i}=2F^{-1}_Z(\tau_i)-F^{-1}_Z(\hat\tau_i)-F^{-1}_Z(\hat\tau_{i+1})
\end{align}
$$

where $$F^{-1}_Z$$ is the quantile function. Because now $$\tau_i$$s no longer uniformly spread between $$0$$ and $$1$$, we change the action-value function accordingly

$$
\begin{align}
Q(s,a)=\sum_{i=0}^{N-1}(\tau_{i+1}-\tau_i)F_Z^{-1}(\hat\tau_i)
\end{align}
$$


[Elevator back to directory](#dir)

### <a name="rainbow"></a>[Rainbow]({{ site.baseurl }}{% post_url 2018-10-27-Rainbow %})

Rainbow combines 6 improvements on DQN:

1. Distributional DQN
2. Multi-step learning
3. DDQN
4. Dueling DQN
5. PER
6. Noisy Networks

#### [Double DQN]({{ site.baseurl }}{% post_url 2018-10-27-Rainbow %})

DQN suffers the overestimation bias introduced by two facts: 1) $$Q$$-function is noisy, and 2) the expected value of the maximum of a set of variables is greater than or equal to the maximum of their expected values, i.e.,

$$
\begin{align}
r(s,a)+\gamma \mathbb E\left[\max_{a'}Q^-(s',a')\right]\ge r(s,a)+\gamma\max_{a'}\mathbb E[Q^-(s',a')]
\end{align}
$$

which is obtained from Jensen's inequality and the convex property of $$\max$$.

Double DQN(DDQN) mitigates the overestimation bias of DQN by having the online network pick the best action for the next state, which gives the target value as

$$
\begin{align}
r(s,a)+\gamma Q^-\left(s',\underset{a'}{\arg\max}Q(s',a')\right)
\end{align}
$$


[Elevator back to directory](#dir)

#### <a name="duel"></a>[Dueling DQN]({{ site.baseurl }}{% post_url 2018-10-27-Rainbow %})

Dueling DQN divides $$Q$$-function into value and advantage functions:

$$
\begin{align}
Q(s,a)=V(f(s))+A(f(s),a)-{\sum_{a'}A(f(s),a')\over |\mathcal A|}
\end{align}
$$

The last term is introduced since it is not sufficient, with only the first two terms, to uniquely recover $$V$$ and $$A$$ for a given $$Q$$. If we replace the last term with a maximum, then $$V$$ can be nicely interpreted as the optimal action-value $$\max_{a}Q(s,a)$$. The reason we use the average is to increase the stability of the optimization.

[Elevator back to directory](#dir)

#### <a name="noisy"></a>[Noisy Networks]({{ site.baseurl }}{% post_url 2018-10-27-Rainbow %})

Noisy networks introduces extra layers in addition to traditional linear layers to add noise to the parameters, whereby providing some explorations:

$$
\begin{align}
y&=(W+W_{noisy}\cdot\epsilon^w)x+(b+b_{noisy}\cdot\epsilon^b)\\\
&=(Wx+b)+((W_{noisy}\cdot\epsilon^w)x+b_{noisy}\cdot\epsilon^b)
\end{align}
$$


[Elevator back to directory](#dir)

#### <a name="per"></a>[Prioritized Experience Replay]({{ site.baseurl }}{% post_url 2018-10-14-PER %})

Prioritized Experience Replay(PER) samples transitions with probability $$P(i)$$ proportional to their priority $$p_i^\alpha$$, where $$\alpha$$ determines how much prioritization is used. Two variants are proposed: 

1. Proportional prioritization: a sum-tree, $$p_i=\vert \delta_i\vert +\epsilon$$
2. Rank-based prioritization: a priority queue, $$p_i={1\over\mathrm{rank}(i)}$$.

To correct bias towards transitions of high priority, we apply weighted importance sampling, which results in the loss

$$
\begin{align}
\mathcal L&={1\over \max_i w_i}w_i\delta_i^2\\\
w_i&=\left({1\over N}{1\over P(i)}\right)^\beta
\end{align}
$$

where $$1\over \max_iw_i$$ normalizes ratios to avoid scaling the update upward, $$\beta$$ in general anneals from its initial value to $$1$$ towards the end of learning.

[Elevator back to directory](#dir)

### <a name="popart"></a>[PopArt]({{ site.baseurl }}{% post_url 2019-10-07-PopArt %})

PopArt deals with rewards of different scales using the running mean $$\mu$$ and scale $$\Sigma$$ of the target $$Y$$.

[Elevator back to directory](#dir)

### <a name="tbo"></a>[Transformed Bellman Operator]({{ site.baseurl }}{% post_url 2020-01-01-Apex-DQfQ %})

TBO defines the following operator to address rewards of large scale

$$
\begin{align}
\mathcal T_h(Q)(x,a)&:=\mathbb E_{x'\sim P(\cdot|x,a)}\left[h\left(R(x,a)+\gamma\max_{a'\in\mathcal A}h^{-1}(Q(x',a'))\right)\right]\\\
where\quad h(z)&=\text{sign}(z)(\sqrt{|z|+1}-1)+\epsilon z, \epsilon=10^{-2}
\end{align}
$$

$$h(z)$$ is chosen because both $$h(z)$$ and $$h(z)^{-1}$$ are monotonically increasing and Lipschitz continuous.

[Elevator back to directory](#dir)

### <a name="retrace"></a> [Retrace]({{ site.baseurl }}{% post_url 2020-11-07-Retrace %})

Retrace defines the following truncated return-based target for off-policy learning

$$
\begin{align}
\mathcal RQ(x,a)=&Q(x,a)+\mathbb E_\mu\left[\sum_{k=t}^{t+n-1}\gamma^{k-t}\left(\prod_{i=t+1}^{k}c_i\right)\delta_k\right]\\\
where\quad c_{i}=&\lambda\min\left(1,{\pi(a_i|x_{i})\over\mu(a_i|x_i)}\right)\\\
\delta_k=&r_k+\gamma\mathbb E_{\pi}Q(x_{k+1},\cdot)-Q(x_{k},a_{k})
\end{align}
$$

Retrace enjoys the following advantages

1. Retrace does not suffer from variance explosion as importance ratios are truncated.
2. It does not cut in on-policy case, making it possible to benefit from the full returns
3. In the evaluation setting, Retrace is a $$\gamma$$-contraction around $$Q^{\pi}$$ for arbitrary policies $$\mu$$ and $$\pi$$.
4. In the control setting (where the behavior policies are increasingly greedy), Retrace converges almost surely to $$Q^*$$, without requiring the GLIE assumption.

[Elevator back to directory](#dir)

### <a name="r2d2"></a>[R2D2]({{ site.baseurl }}{% post_url 2019-11-21-R2D2 %})

R2D2 trains an LSTM $$Q$$-network with off-policy data. To mitigate the state-shift issue, R2D2 stores the LSTM state in the replay and replay a portion of sequence(burn-in) before training.

[Elevator back to directory](#dir)

### <a name="agent57"></a>[Agent57]({{ site.baseurl }}{% post_url 2020-05-01-Agent57 %})

Agent57, built upon NGU, learns two universal $$Q$$-functions for the intrinsic and extrinsic rewards, respectively. To maintain high performance on dense reward environment, it introduces a meta-controller to each actor to select which $$Q(i)$$ is used for data collection at the beginning of each episode, where $$i$$ indexes the $$i^{th}$$ $(\gamma,\beta)$$ pair. 

[Elevator back to directory](#dir)

### <a name="decisions_dqn"></a>Decisions in DQN

[Hasselt et al. 2018]({{ site.baseurl }}{% post_url 2020-07-27-deadly-triad %}) conduct a large scale study on Atari57 and find that

1. The divergence of the $$Q$ function often results in poor performance 
   1. The target network, double Q, and multi-step returns can effectively reduce the occurrence of divergence and improve the performance
   2.  Strong prioritization can lead to instability
2. A large network is more likely to increase divergence, but still yields better performance. One hypothesis is that when the network is large, the update of the value function may not be generalized to the next state, which helps stabilize the target value.

[Fu et al. 2019]({{ site.baseurl }}{% post_url 2019-12-01-DQN %}) find that

1. A large network not only is crucial for representing a better solution but also make it easier to train using bootstrapping.
2. It is less likely to overfit when training with data of high entropy. This stresses the importance of the experience replay in value-based algorithms

## <a name="dpg"></a> Deterministic Policy Gradient

Although deterministic policy gradient algorithms use an actor for action selection, they are more likely to fail in the category of value-based RL instead of policy-gradient RL as the policy is guided by the $$Q$$-function and the quality of $$Q$$-function plays an crucial role in the performance of the algorithms.

### <a name="ddpg"></a>[Deep Deterministic Policy Gradient]({{ site.baseurl }}{% post_url 2018-10-07-PG %})

In Deep Deterministic Policy Gradient(DDPG), policy($$\mu(s)$$) deterministically selects an action that maximize the $$Q$$-value. DDPG uses target networks and experience replay to stabilize deep neural network training as DQN does.

[Elevator back to directory](#dir)

### <a name="td3"></a>Twin Delayed Deep Deterministic Policy Gradient Algorithm

Twin Delayed Deep Deterministic policy gradient algorithm(TD3) addresses the overestimation bias problem in DDPG by simultaneously training two $$Q$$-functions. In TD3, we always have the actor maximizes $$Q_1$$ but define the target $$Q$$-value as

$$
\begin{align}
r(s,a)+\gamma\min_{i=1, 2}Q^-_{i}(s',\mu^-(s')+\epsilon)
\end{align}
$$


where a small noise $$\epsilon$$ is added to $$\mu^{-}(s')$$ to regularize the $$Q$$ functions, making them robust to small errors. 

[Elevator back to directory](#dir)

### <a name="d4pg"></a>Distributed Distributional DDPG

Distributed Distributional DDPG(D4PG) combines four improvements on DDPG:

1. Ape-X
2. Distributional DQN
3. Multi-step learning
4. PER

[Elevator back to directory](#dir)
