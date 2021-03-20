---
title: "Challenges of Real-World Reinforcement Learning"
excerpt: "Discussion on several challenges of real-world reinforcement learning."
categories:
  - Reinforcement Learning
tags:
  - Overviews
---

## Challenges

Dulac-Arnold et al. enumerate several challenges of real-world reinforcement learning:

1. Training offline from the fixed logs of an external behavior policy.
2. Learning on the real system from limited samples.
3. High-dimensional continuous state and action spaces.
4. Safety constraints that should never or at least rarely be violated.
5. Tasks that may be partially observable, alternatively viewed as non-stationary or stochastic.
6. Reward functions that are unspecified, multi-objective, or resk-sensitive.
7. System operators who desire explainable policies and actions.
8. Inference that must happen in real-time at the control frequency of the system.
9. Large and/or unknown delays in the system actuators, sensors, or rewards.

## <a name="brl"></a>Batch Reinforcement Learning

We have covered several algorithms in batch reinforcement learning(BRL) in the previous posts[[1]({{ site.baseurl }}{% post_url 2019-12-07-BCQ %}), [2]({{ site.baseurl }}{% post_url 2019-12-14-REM %})], another important aspect of BRL is off-policy evaluation, which refers to evaluate the policy's performance without running it on the real system. There are three approaches to off-policy evaluation:

1. **Importance sampling:** Evaluate the target policy using
   
$$
   G_{\pi}=\mathbb E_{b}\left[\omega_{0:T}G_b\right]
   $$

   

   where $$G_\pi,G_b$$ are the cumulative rewards under the target and behavior policy, and $$\omega_{0:T}=\prod_{t=0}^T{\pi(a_t\vert s_t)\over b(a_i\vert s_i)}$$. As IS is notoriously known as high variance, one could reduce its variance using weighted importance sampling(WIS) and causality, which gives
   
$$
   G_{\pi}= \sum_{i=1}^n{\omega_{0:t}^{(i)}\over\sum_{i=1}^n\omega_{0:t}^{(i)}}\sum_{t=0}^T\gamma^{t}\omega_{0:t}^{(i)}r_t
   $$


2. **Direct Method:** Learn a transition or value function and use that for evaluation. The difficulty of this method is that there is no clear method to quantify the approximation error or bias.

3. **Doubly robust estiamtor:** Combine both methods using.
   
$$
   G_\pi=\mathbb E_b\left[\sum_{t=0}^T\gamma^t\left(\omega_{0:t}r_t-(\omega_{0:t}Q(s_t,a_t)-\omega_{0:t-1}V(s_t,a_t))\right)\right]
   $$

   here we do not apply WIS for simplicity. Advanced methods include MAGIC(Thomas & Brunskill, 2016) and more robust doubly robust (Farajtabar et al. 2018) 

## Learning on The Real System from Limited Samples

Learning on the real system generally requires an algorithm to be sample-efficient. As a result, meta-learning algorithms or model-based algorithms are considered as a promising approaches.

## High-Dimensional Continuous State and Action Spaces

Dulac-Arnold et al. propose evaluating policy performance for high-dimensional continuous action spaces according to two dimension: number of actions, and relation over the action. Millions of related actions are much easier to learn than a couple hundred completely unrelated actions, as the agent needs to sufficiently sample each individual action in the latter case.

## Satisfying Safety Constraints

Several work in RL safety ([Dalal et al. 2018](#dalal2018)) has cast safety in the context of Constraint MDPs(CMDPs), which defines a constrainted optimization problem:

$$
\begin{align}
\max_\pi R(\pi)\\\
s.t.\quad C^k(\pi)\le V_k,k=1,\dots, K
\end{align}
$$

where $$R$$ is the cumulative reward of a policy $$\pi$$, $$C^k(\pi)$$ describes the incurred cumulative cost of a certain policy $$\pi$$ relative to constraint $$k$$, and $$V_k$$ is the constriant level

An alternative to CMDPs is budgeted MDPs([Carrara et al. 2018](#carrara2018)), where the policy is learned as a function of constriant level. This allows the user to examine the trade-offs between expected return and constraint level and choose the constraint level that best works for the data.

## Partial Observability and Non-Stationary

There are two common approaches to handle partial observability in the literature. First is to incorporate history into the observation of the agent. For example, DQN stacks four Atari frames together as the agent's observation to account for partial observability. An alternative approach is to use recurrent networks within the agent. [Hausknecht & Stone](#hausknetcht2015) apply RNN to DQN, and show that the recurrent version can perform equally well in Atari games when only given a single frame as input.

Domain randomization ([Peng et al. 2018](#peng2018)) involves explicitly training an agent on various perturbations of the environment and averaging these learning errors together during training. System identification involves training a policy that can determine online the environment in which it is operating and modify the policy accordingly.

## Unspecified and Multi-Objective Reward Function

In some cases, the agent is asked to perform well for all task instances not just in expectation. Therefore, policy quality cannot be summarized by a single scalar describing cumulative reward, but must consider both multiple dimensions of the policy's behavior and the full distribution of behaviors.

A typical approach to evaluate the full distribution of reward across groups is to use a Conditional Value at Risk (CVaR) objective ([Tamar et al. 2015](tamar2015)), which looks at a given percentile of the reward distribution, rather than expected reward. Tamar et al. show that by optimizing reward percentiles, the agent is able to improve oupon its worst-case performance.

Van Seijen et al. present an approach that takes advantage of multi-objective reward signals to learn super-human performance in the Atari game Mc-PacMan.

## System Delay

[Hung et al. 2018](#hung2018) introduce a method to better assign rewards that arrive signiﬁcantly after a causative event. They use a memory-based agent, and leverage the memory retrieval system to properly allocate credit to distant past events that are useful in predicting the value function in the current timestep. They show that this mechanism is able to solve previously unsolveable delayed reward tasks.

## References

<a name="dalal2018"></a>Dalal, G., Dvijotham, K., Vecerik, M., Hester, T., Paduraru, C., and Tassa, Y. Safe exploration in continuous action spaces. CoRR, abs/1801.08757, 2018. URL http://arxiv.org/abs/1801.08757.

<a name="carrara2018"></a>Carrara, N., Laroche, R., Bouraoui, J., Urvoy, T., Olivier, T. D. S., and Pietquin, O. A ﬁtted-q algorithm for budgeted mdps. In EWRL 2018, 2018.

<a name="hausknetcht2015"></a>Hausknecht, M. J. and Stone, P. Deep recurrent q-learning for partially observable mdps. CoRR, abs/1507.06527, 2015

<a name="peng2018"></a>Peng, X. B., Andrychowicz, M., Zaremba, W., and Abbeel, P. Sim-to-real transfer of robotic control with dynamics randomization. In 2018 IEEE International Conference on Robotics and Automation (ICRA), pp. 1–8. IEEE, 2018

<a name="tamar2015"></a>Tamar, A., Glassner, Y., and Mannor, S. Optimizing the cvar via sampling. In Twenty-Ninth AAAI Conference on Artiﬁcial Intelligence, 2015b.

<a name="seijen2017"></a>Harm van Seijen, Mehdi Fatemi, Joshua Romoff, Romain Laroche, Tavian Barnes, Jeffrey Tsang. Hybrid Reward Architecture for Reinforcement Learning. in NIPS 2017

<a name="hung2018"></a>Hung, C.-C., Lillicrap, T., Abramson, J., Wu, Y., Mirza, M., Carnevale, F., Ahuja, A., and Wayne, G. Optimizing agent behavior over long time scales by transporting value. arXiv preprint arXiv:1810.06721, 2018.