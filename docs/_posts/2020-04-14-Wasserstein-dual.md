---
title: "From 1st Wasserstein to Kantorovich-Rubinstein Duality"
excerpt: "An introduction to the dual of 1st Wasserstein Distance"
categories:
  - Mathematics
tags:
  - Mathematics
  - Deep Learning
---

## Introduction

In the [previous post]({{ site.baseurl }}{% post_url 2020-04-07-dual %}), we discussed duality in linear programming. In this post, we apply it to minimize the 1st Wasserstein distance($$W_1$$, a.k.a., earth mover's distance). 

## 1st Wasserstein Distance

Let $$(M, d)$$ be a metric space where $$M$$ is a set and $$d(x,y)=\vert x-y\vert $$ is a distance function/metric on $$M$$. The 1st Wasserstein distance between two probability measure $$\mu$$ and $$\nu$$ is defined as

$$
W_1(\mu,\nu)=\inf_{\gamma\in\Gamma(\mu,\nu)}\int_{M\times M}d(x,y)d\gamma(x,y)
$$

where $$\Gamma(\mu,\nu)$$ denotes the set of all coupling of $$\mu$$ and $$\nu$$. If we regard 1st Wasserstein distance as the earth mover's distance, we can interpret $$\mu$$ and $$\nu$$ as two piles of dirt. Therefore, $$d(x, y)$$ becomes the cost of moving from $$x$$ to $$y$$, $$\gamma(x,y)$$ denotes a transport plan that moves the amount of dirt from $$x$$ to $$y$$, and $$W_1(\mu,\nu)$$ is the minimum cost of turning $$\mu$$ into $$\nu$$. 

## Formulating 1st Wasserstein Distance as a Dual Problem

To meet our purpose, we define the primal problem as a minimization problem:

$$
\begin{align}
&\text{minimize}&\pmb c^\top\pmb x\\\
&s.t.&\pmb A\pmb x=\pmb b\\\
&&\pmb x\ge 0
\end{align}
$$

where $$\pmb x=vec(\Gamma)$$ is a flatten vector of $$\Gamma$$, $$\pmb c=vec(D)$$ is a flatten vector of the set of distance $$D=\{d(x,y)\vert x,y\in M\}$$, $$\pmb b=\begin{bmatrix}\pmb \mu\\\\pmb \nu\end{bmatrix}$$ is the concatenation of marginals $$\pmb \mu$$ and $$\pmb \nu$$(we use the bold font to denote values of the corresponding functions), and $$\pmb A$$ is a sparse binary matrix such that $$\pmb A[i]\pmb x=\pmb b[i]$$, which enforces the marginal constraints $$\int\gamma(x,y)dy=\mu(x)$$ and $$\int\gamma(x,y)dx=\nu(y)$$. It may be clearer to divide $$\pmb A$$ into two matrices, where we have $$\pmb A_1\pmb x=\pmb \mu$$ and $$\pmb A_2\pmb x=\pmb \nu$$. For $$\pmb A_1$$ and $$\pmb A_2$$, each column corresponds to a transfer plan in $$\pmb x$$ and there is exactly one $$1$$ in each column as transport plans are only counted once. For example, $$\pmb x[k]$$ is a tranport plan from $$x$$ to $$y$$, then $$\pmb A_1[x,k]=1$$ and $$\pmb A_2[y,k]=1$$. Therefore, there are two $$1$$s in each column of $$\pmb A$$. This property will come in handy next when we discuss  the dual.

The dual is a maximization problem defined as

$$
\begin{align}
&\text{maximize}&\pmb b^\top\pmb y\\\
&s.t.&\pmb A^\top\pmb y\le\pmb c
\end{align}
$$

Notice that the sign constraints on $$\pmb y$$ is removed as the primal constraints becomes equality(see the proof in supplementary for details).

Like $$\pmb b=\begin{bmatrix}\pmb \mu\\\\pmb \nu\end{bmatrix}$$, let $$\pmb y=\begin{bmatrix}\pmb f\\\\pmb g\end{bmatrix}$$. Following our previous analysis on $$\pmb A$$, we can see that, for each constraint $$\pmb c[k]=d(x,y)$$, we have $$\pmb A_1^\top[k,x]\pmb f[x] + \pmb A_2^\top[k,y]\pmb g[y]=\pmb f[x]+\pmb g[y]\le d(x,y)$$. Now we rewrite the dual as 

$$
\begin{align}
&\text{maximize}&\pmb \mu^\top \pmb f+\pmb \nu^\top \pmb g&\\\
&s.t.&\pmb f[x]+\pmb g[y]\le d(x,y)&\quad \forall x,y
\end{align}
$$

##Applying the Lipschitz constraint

We can also define the function version of the above dual, usually so-called Kantorovich duality, as

$$
\sup_{f(x)+g(y)\le d(x,y)}\int f d\mu(x)+\int gd\nu(y)\tag {1}
$$

From now on, we stick to this function version as it's more general and more easy to analyze.

Let's assume we have a function $$f$$. It is easy to conclude from the constraint that the supremum defined in $$(1)$$ is achieved when $$g(y)=\inf_x d(x,y) - f(x)$$ since $$d\mu, d\nu\ge 0$$. This function is often called $$c$$-transform and denoted by $$f^c(y)=g(y)=\inf_x d(x,y) - f(x)$$. Replacing $$g$$ with $$f^c$$, we rewrite Equation $$(1)$$ as

$$
\sup_f\int fd\mu(x)+\int f^cd\nu(y)\tag {2}
$$

An interesting property of $$f^c$$ is that when $$f$$ is 1-Lipschitz, $$f^c$$ is Lipschitz too as $$d(x, y)=\vert x-y\vert $$ is 1-Lipschitz. For all $$x$$ and $$y$$, when $$f$$ is 1-Lipchitz this gives us

$$
\begin{align}
&|f^c(y)-f^c(x)|\le|y-x|\\\
\Rightarrow&-f^c(x)\le |y-x|-f^c(y)\\\
\Rightarrow&-f^c(x)\le \inf_y|y-x|-f^c(y)\le -f^c(x)
\end{align}
$$

where the last inequality holds by choosing $$y=x$$ in the infimum. This gives us $$-f^c(x)=\inf_y\vert y-x\vert -f^c(y)$$. Also, noticing that $$f^c(y)=g(y)$$ and $$f(x)=\inf_y\vert y-x\vert -g(y)$$(obtained through the same $$c$$-transform), we get $$f(x)=-f^c(x)$$. Substituting $$f^c(x)=-f(x)$$ in Equation $$(2)$$, we get

$$
\sup_{f\text{ is Lipschitz}}\int f(d\mu-d\nu)
$$

which is the dual form of 1-Wasserstein distance discussed in the WGAN paper.

## Why Wasserstein is preferred over KL and JS distance?

The simple answer is that Wasserstein distance is smoother than KL and JS distance. Consider two distributions, $$P$$ and $$Q$$, 

$$
\begin{align}
P(x,y)&=
\begin{cases}
1&\text{if }x=0, y\in[0,1]\\\
0&\text{otherwise}
\end{cases}\\\
Q(x,y)&=
\begin{cases}
1&\text{if }x=\theta, y\in[0,1]\\\
0&\text{otherwise}
\end{cases}
\end{align}
$$

When $$\theta\ne 0$$, we have 

$$
\begin{align}
D_{KL}(P,Q)&=+\infty\\\
D_{KL}(Q,P)&=+\infty\\\
D_{JS}(P,Q)&={1\over 2}\left(\int_{x=0,y\in[0,1]}1\cdot\log{1\over {.5}}dxdy
+\int_{x=\theta,y\in[0,1]}1\cdot\log{1\over {.5}}dxdy\right) = \log 2\\\
W_1(P,Q)&=|\theta|
\end{align}
$$

When $$\theta=0$$, the two distributions are overlapped and all of the above metrics are zero. We can see that the KL divergence blows up to infinity when two distributions are disjoint, while the JS divergence stays constant when two distribution are disjoint and suddenly drop to zero once they overlapped. Only the Wasserstein is continuous and smooth.

## References

http://abdulfatir.com/Wasserstein-Distance/

https://www.youtube.com/watch?v=1ZiP_7kmIoc

## Supplementary

### Proof of Strong Duality

Following the similar process in the previous post, let 

$$
\pmb{\hat A}=\begin{bmatrix}\pmb A\\\-\pmb c^\top\end{bmatrix},\pmb{\hat b}=\begin{bmatrix}\pmb b\\\-(\tau-\epsilon)\end{bmatrix}
$$

where $$\tau=\pmb c^\top\pmb x^*$$ is the optimal value of the primal, $$\epsilon\ge 0$$ is an arbitrary value. Because $$\tau$$ is the optimal value, there is no feasible $$\pmb x$$ such that $$\pmb c^\top\pmb x=\tau-\epsilon$$. Therefore, there is no $$\pmb x\in\mathbb R^n$$ such that

$$
\begin{bmatrix}\pmb A\\\-\pmb c^\top\end{bmatrix}\pmb x= \begin{bmatrix}\pmb b\\\-(\tau+\epsilon)\end{bmatrix}
$$

By Farkas' Lemma, there exists $$\pmb{\hat y}=\begin{bmatrix}\pmb y\\\ \alpha\end{bmatrix} $$, such that

$$
\begin{bmatrix}\pmb A^\top&-\pmb c\end{bmatrix}\begin{bmatrix}\pmb y\\\\alpha\end{bmatrix}\le\pmb 0,
\quad \begin{bmatrix}\pmb b^\top&-(\tau+\epsilon)\end{bmatrix}\begin{bmatrix}\pmb y\\\\alpha\end{bmatrix}>0\\\
$$

Thus, we have

$$
\pmb A^\top\pmb y\le\alpha\pmb c,\quad \pmb b^\top\pmb y>\alpha(\tau+\epsilon)
$$

If $$\alpha=0$$, then the primal is infeasible -- Because of $$\pmb A^\top\pmb y\le\pmb 0$$ and $$\pmb b^\top\pmb y>0$$, by Farkas' Lemma, there is no $$\pmb x$$ satisfying the primal constraints. Therefore, $$\alpha>0$$ and by scaling $$\pmb {\hat y}$$, we may assume that $$\alpha=1$$. So $$\pmb A^\top \pmb y\le \pmb c$$ and $$\pmb b^\top\pmb y\le \tau+\epsilon$$. By the Weak Dual Theorem, we have $$\tau= \pmb c^\top \pmb x\le\pmb b^\top\pmb y\le\tau+\epsilon$$. Since $$\epsilon$$ is arbitrary, we have $$\pmb c^\top \pmb x=\pmb b^\top\pmb y$$.

### Another Way of Deriving Kantorovich Duality

Consider the 1st Wasserstein distance 

$$
W_1(\mu,\nu)=\inf_{\gamma\in\Gamma(\mu,\nu)}\int_{M\times M}d(x,y)d\gamma(x,y)
$$

We can remove the constraint by adding an additional term

$$
W_1(\mu,\nu)=\inf_\gamma\int_{M\times M}d(x,y)d\gamma(x,y)+\underbrace{\sup_{f,g}\int fd\mu(x)+\int gd\nu-\int(f(x)+g(y))d\gamma(x,y)}_{=\begin{cases}0&\text{if }\gamma\in\Gamma(\mu,\nu)\\\\infty&\text{otherwise}\end{cases}}
$$

The supremum term impose the constraint $$\gamma\in\Gamma(\mu,\nu)$$: it's $$0$$ when the constraint is satisfied, and otherwise we can choose suitable $$f$$ and $$g$$ to make it infinity. 

Now we move the $$\sup_{f,g}$$ outside because the first term does not depend on $$f$$ and $$g$$

$$
W_1(\mu,\nu)=\inf_\gamma\sup_{f,g}\int_{M\times M}d(x,y)d\gamma(x,y)+\int fd\mu(x)+\int gd\nu(y)-\int(f(x)+g(y))d\gamma(x,y)
$$

Assuming the minimax principle, we swap $$\inf$$ and $$\sup$$

$$
W_1(\mu,\nu)=\sup_{f,g}\inf_\gamma\int_{M\times M}d(x,y)-(f(x)+g(y))d\gamma(x,y)+\int fd\mu(x)+\int gd\nu(y)
$$

Take a look at the infimum term. If $$d(x,y)-f(x)+g(y)$$ could be negative, then we can choose $$\gamma$$ such that an infinity mass is on that negative value and the infimum term becomes $$-\infty$$. On the other hand, if $$d(x,y)-f(x)+g(y)$$ is non-negative, then the infimum term is $$0$$. From the above observation, we can summarize the infimum term as a constraint and the 1st Wasserstein distance becomes Equation $$(1)$$

$$
W_1(\mu,\nu)=\sup_{f(x)+g(y)<\gamma(x,y)}\int fd\mu(x)+\int gd\nu(y)
$$

