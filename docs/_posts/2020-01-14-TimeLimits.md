---
title: "Time Limits in Reinforcement Learning"
excerpt: "In which we discuss the impact of time limits in reinforcement learning"
categories:
  - Reinforcement Learning
tags:
---

## Introduction

We discuss time limits in reinforcement learning, which I overlooked before but recently found it could significantly affect the agent's performance in a variety of environments. 

## The Most Important

By default, many environments, such as those from openAI's gym, emit done signals either when it reaches the terminal state or when time's out. The later is problematic for tasks without a time limit in nature as it introduces inconsistent expected returns for similar states at different time step. Therefore, it should be manually corrected for those cases.

<div style="text-align: right"> A bitter lesson learned from a wasted month:-( </div>

## Time Limits for Fixed-Period Tasks

For fixed-period tasks, time limits are in fact a part of the environment. According to Pardo et al.'s work, incorporating the remaining time into the agent's inputs can ease the problem of state aliasing -- similar states take on different credit at different time -- which generally leads to suboptimal policies and instability due to infeasibility of correct credit assignment. 

## Time Limits for Indefinite-Period Tasks

For indefinite-period tasks, time limits are generally used to introduce diversities to environment, thereby facilitating learning. The results reported by Pardo et al. are consistent with the observation we discussed in section "The Most Important". 

Another interesting observation is that, with partial-episode bootstrapping(PEB), the performance of an agent with experience replay becomes insensitive to the size of the replay buffer, which is previously studied by Zhang & Sutton.

## Encoding Time

Pardo et al. represent the remaining time as a scalar in range of `[-1, 1]`. 

## References

Fabio Pardo, Arash Tavakoli, Vitaly Levdik, and Petar Kormushev. Time Limits in Reinforcement Learning

Martha White. Unifying task speciﬁcation in reinforcement learning.

Shangtong Zhang, Richard S, Sutton. A deeper look at experience replay.