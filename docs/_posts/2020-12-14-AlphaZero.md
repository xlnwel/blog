---
title: "AlphaZero"
excerpt: "In which we discuss AlphaZero, an agent that achieves super-human performance in chess, shogi and Go"
categories:
  - Reinforcement Learning
tags:
  - Multi-Agent Reinforcement Learning
  - Distributed Reinforcement Learning
  - Reinforcement Learning Application
  - Model-Based Reinforcement Learning
---

## Introduction

We briefly discuss AlphaZero without mentioning any game-specific details. We will discuss more in the next post when we talk about MuZero, a successor of AlphaZero. 

## Overview

<figure>
  <img src="{{ '/images/application/AlphaZero-CheetSheet.png' | absolute_url }}" alt="" width="1000">
  <figcaption>A cheat sheet of AlphaGo Zero, the precessor of AlphaZero that is designed for game Go</figcaption>
  <style>
    figure figcaption {
    text-align: center;
    }
  </style>
</figure>

At each time step $$t$$, AlphaZero runs [Monte Carlo tree search]({{ site.baseurl }}{% post_url 2018-11-14-planning %})(MCTS) from the current position $$s_t$$ and selects an action based on the root visit count, either proportionally (for exploration) or greedily (for exploitation). In the tree, each node represents a state produced by the simulator, and each edge (a state-action pair) stores a set of statistics, $$\{N(s,a), W(s,a), Q(s,a), P(s,a)\}$$, where $$N(s,a)$$ is the visit count, $$W(s,a)$$ is the total action-value, $$Q(s,a)$$ is the mean action-value, and $$P(s,a)$$ is the prior probability of selecting $$a$$ in $$s$$. The tree expands as traditional MCTS except that

1. We use the upper confidence bound defined as $$Q(s,a)+C(s)P(s,a)\sqrt{N(s)}/(1+N(s,a))$$, where $$N(s)$$ is the parent visit count and $$C(s)$$ is the exploration rate, which grows slowly with search time, $$C(s)=\log((1+N(s)+c_{base})/c_{base})+c_{init}$$. 
2. [Dirichlet]({{ site.baseurl }}{% post_url 2019-04-01-discrete-probability-distribution %}) noise is added to the prior probabilities in the root node $$s_0$$, specifically $$P(s_0,a)=(1-\epsilon)p_a+\epsilon\eta_a$$, where $$\eta\sim \text{Dir}(\alpha)$$ and $$\epsilon=0.25$$. This is done at the start of each move before any MCTS to increase the amount of searches to moves that looks like bad with a shallow search but reveals to be good with a deeper search.
3. AlphaZero trains a value network $$v$$ to indicate the value of a leaf state rather than running simulated trajectories. 

When a simulation reaches a leaf node $$s_L$$, we initialize all its state-action pair $$(s_L, a)$$ to $$\{N(s_L,a)=0, W(s_L,a)=, Q(s_L,a)=0, P(s_L,a)=p_a\}$$, where $$p_a$$ is a policy network. The visit counts and values are then updated in a backward pass through each step $$t\le L$$, 

$$
\begin{align}
N(s_t,a_t)&=N(s_t,a_t)+1\\\
W(s_t,a_t)&=W(s_t,a_t)+v\\\
Q(s_t,a_t)&={W(s_t,a_t)\over N(s_t,a_t)}
\end{align}
$$

## Network Training

AlphaZero trains networks through self-play, in which the agent always plays with the latest version of itself -- this is different from AlphaGo which play agaist the current best model). At the end of the game, the terminal position $$s_T$$ is scored by the game outcome $$z$$: $$-1$$ for a loss, $$0$$ for a draw, and $$+1$$ for a win. $$v$$ is trained to minimize the mean squared error between $$v$$ and $$z$$, $$p$$ is trained to minimize the KL divergence between $$p$$ and search tree policy $$\pi$$. Both $$v$$ and $$p$$ share the convolutionary layers, regularized by $$\ell_2$$ regularization. We summarize the loss function as

$$
\begin{align}
\mathcal L=(z-v)^2-\pi\log\ p + c\Vert\theta\Vert^2
\end{align}
$$

where $$c$$ is a hyperparameter controlling the level of regularization.

## Comparison with Alpha-Beta Search

It is worth noting that AlphaZero uses a combination of MCTS and deep neural networks, while previous chess programs often adopt alpha-beta search and a linear evaluation function. A deep neural network provides a more powerful evaluation function, but may also introduce larger worst-case generalization errors. When combined with alpha-beta search, which computes an explicit minimax, the biggest errors are typically propagated directly to the root of the subtree. By contrast, MCTS averages over the position evaluations within the subtree, rather than computing the minimax evaluation of that subtree. The approximation errors introduced by neural networks therefore tends to be cancel out when evaluating a large subtree.

## References

Silver, David, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot, et al. 2018. “A General Reinforcement Learning Algorithm That Masters Chess, Shogi, and Go through Self-Play.” *Science* 362 (6419): 1140–44. https://doi.org/10.1126/science.aar6404.