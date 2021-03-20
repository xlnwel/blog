---
title: "Basic Policies in Reinforcement Learning"
excerpt: "We talk in detail about some wildly used policy in reinforcement learning, including epsilon-greedy policy, softmax with temperature, upper confidence bound(UCB), and gradient bandit algorithm"
categories:
  - Reinforcement Learning
tags:
  - Exploration in RL
image: 
  path: /images/rl/Policy.png
---

## Introduction

This post talks about several policies wildly used in reinforcement learning and explains some intuitions behind them to help fully understanding.

## <a name="dir"></a>Table of Contents

- [Epsilon-greedy Policy](#epsilon)
- [Softmax with Temperature](#soft)
- [Upper Confidence Bound](#ucb)
- [Gradient Bandit Algorithm](#gba)

## <a name="epsilon"></a>Epsilon-greedy Policy

Epsilon-greedy policy simply follows a greedy policy with probability $$ 1-\epsilon $$ and takes a random action with proabability $$ \epsilon $$. Formally, it's defined as

$$
\begin{align}
\pi(a|s)=(1-\epsilon)_{|a=\arg\max_{a'}}Q(s,a')+{\epsilon\over |A|}
\end{align}
$$


 A further refinement is to start with $$ \epsilon=1 $$, and then gradually decay as the time goes by. At last, $$ \epsilon $$ will stop at some small value.

[Elevator back to directory](#dir)

## <a name="soft"></a>Softmax with Temperature

Softmax with temperature refines simply softmax to encourage exploration at the beginning and enhance the convergence at the end. Mathematically, It's defined as

$$
\begin{align}
\pi(a|s)=\mathrm{softmax}(Q(s,a)/\tau)={\exp(Q(s,a)/\tau)\over\sum_{a'}\exp(Q(s,a')/\tau)}
\end{align}
$$

where the temporature $$ \tau $$, idea borrowed from *simulated annealing*, is annealed over time. A high temporature causes all actions equiprobable, while a low temporature skews the probability towards a greedy policy.  

[Elevator back to directory](#dir)

## <a name="ucb"></a>Upper Confidence Bound

Upper Confidence Bound (UBC) encourages actions rarely tried by introducing a concept of upper confidence bound. Mathematically, it's defined as

$$
\begin{align}
\pi(a|s)=\arg\max_a\left(Q(s,a)+c\sqrt{\log t\over N(s,a)}\right)
\end{align}
$$

where $$ t $$ counts the number of times $$ s $$ is visited and $$ N(s, a) $$ the number of times $$ a $$ is selected at the state $$ s $$. The second term above defines a measure of uncertainty or variance in the estimate of the state-action value. The quantity being max’ed over is thus a sort of upper confidence bound on the possible true value of action $$ a $$, with $$ c $$ determining the confidence level. Each time $$ a $$ is selected, both $$ t $$ and $$ N(s,a) $$ increase, the uncertainty is presumably reduced. On the other hand, each time an action other than $$ a $$ is selected, only $$ t $$ increase, the uncertainty estimate increases. The use of the natural logarithm ensures that the increases get smaller over time, but are unbounded; all actions will eventually be selected, but actions with lower value estimates, or that have already been selected frequently, will be selected with decreasing frequency over time.

Beyond intuition, let's look deeper into the square-root term. This term is actually derived from the *Hoeffding's inequality*, which is

$$
\begin{align}
P(E[\bar X]\ge \bar X+t)&\le \exp(-2nt^2)
\end{align}
$$

It says that the probability that the true mean of $$ X $$, $$ E[\bar X] $$, is greater than or equal to the sample mean, $$ \bar X $$, plus some value $$ t $$, is less than $$ e^{-2nt^2} $$ where $$ n $$ is the number of samples used to compute $$ \bar X $$. Sticking in our notation, we have

$$
\begin{align}
P\left(\hat Q(s,a)\ge Q(s,a)+U(s,a)\right)\le \exp\left(-2N(s,a) U(s,a)^2\right)
\end{align}
$$

where $$ \hat Q(s,a) $$ is the underlying real action-value function.

Now let the right hand of the inequality to be $$ p $$, we have 

$$
\begin{align}
\exp(-2N(s,a)U(s,a)^2)&= p\\\
U(s,a)&=\sqrt{-{\log p\over 2N(s,a)}}
\end{align}
$$

By setting the maximum probability $$ p=t^{-2} $$ --- that is, we make the upper bound $$ U $$ so that the probability of expected $$ Q $$ being greater than $$ Q $$ plus the upper bound $$ U $$ decays as time $$ t $$ goes by, we successfully acquire the square root in the original expression

[Elevator back to directory](#dir)

## <a name="gba"></a>Gradient Bandit Algorithm

Gradient Bandit Algorithm is an algorithm specialized for bandit problems, in which no state is involved. It defines a numerical *preference* for each action $$ a $$, $$ H(a) $$ and a softmax policy based on that

$$
\begin{align}
\pi(a)=P(A=a)=\mathrm{softmax}(H(a))
\end{align}
$$

At each step, after selecting action $$ a $$ and receiving reward $$ R $$, all the action preference get updated by

$$
\begin{align}
H(A)=H(A)+\alpha(R-\bar R)(\mathbf 1_{|A=a}-\pi(A))
\end{align}
$$

where $$ \bar R $$, the average of all the rewards up to the current time, serves as a baseline with which the reward is compared. The eccentric term, $$ \mathbf 1_{\vert A=a}-\pi(A) $$, is computed from $$ \partial E[R]\over\partial H[A] $$ (details are given in Reinforcement learning: an introduction). An intuition behind it is given below.

At each step, we want to encourage the most recent action a bit to embody how we can optimize the long-term value based on the reward just received. (by encouraging, I don't mean to increase $$ H(a) $$ recklessly, where $$ a $$ is the most recent action. Here I mean to change $$ H(a) $$ to the same direction as $$ R-\bar R $$ suggests. That is $$ H(a) $$ become relatively larger if $$ R-\bar R>0 $$ or relatively smaller if $$ R-\bar R<0 $$. The opposite story works for discouraging). There are to ways to achieve this: 

1. to directly encourage the most recent action $$ a $$. 
2. to indirectly encourage $$ a $$ by discouraging all other actions. In this way, it may not be a good idea to discourage all actions evenly, because that will leave our softmax policy unchanged(as we'll see later, we also discourage the most recent action). To make a distinction, we notice that actions contribute to $$ \bar R $$ according to their proababilities being selected, which makes it a good heuristic to discourage all actions w.r.t. the action distribution given by the policy. 

The last remaining question is why we also want to discourage the most recent action a bit after encouraging it? The reason is that we would not like to indiscriminately encourage the most recent action. Let's assume $$ R-\bar R>0 $$(similar reasoning works for $$ R-\bar R<0 $$). If $$ a $$ is selected frequently, then it shouldn't get much credit for that it contributes a lot to the lower average reward $$ \bar R $$. On the other hand, if $$ a $$ is selected rarely, then it should get more credit since it contributes little to the lower average reward $$ \bar R $$. For an extreme example, let's consider a case where $$ \bar R=1 $$, and the most recent action $$ a $$ results in $$ R=100 $$. If $$ \pi(a)=0.99 $$, we'd like to think that we just got a lucky receiving $$ R=100 $$ via $$ a $$ rather than seriously believing that the average reward received by executing $$ a $$ is around $$ 100 $$. However, if $$ \pi(a)=0.01 $$, we are more likely to conjecture that $$ a $$ would results in a high reward. 

[Elevator back to directory](#dir)