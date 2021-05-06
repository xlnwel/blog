---
title: "From 1st Wasserstein to Kantorovich-Rubinstein Duality"
excerpt: "An introduction to the dual of the 1st Wasserstein distance."
categories:
  - Mathematics
tags:
  - Mathematics
---

## Introduction

We formulate the dual of the 1st Wasserstein distance.

## 1st Wasserstein Distance

Let \\((M, d)\\) be a metric space where \\(M\\) is a set and \\(d(x,y)=\vert x-y\vert \\) be a distance function/metric on \\(M\\). The 1st Wasserstein distance between two probability measure \\(\mu\\) and \\(\nu\\) is defined as

$$
\begin{align}
W_1(\mu,\nu)=\inf_{\gamma\in\Gamma(\mu,\nu)}\int_{M\times M}d(x,y)d\gamma(x,y)\tag 1
\end{align}
$$

where \\(\Gamma(\mu,\nu)\\) denotes the set of all coupling of \\(\mu\\) and \\(\nu\\). If we regard 1st Wasserstein distance as the earth mover's distance, we can interpret \\(\mu\\) and \\(\nu\\) as two piles of dirt. Consequently, \\(d(x, y)\\) becomes the cost of moving from \\(x\\) to \\(y\\), \\(\gamma(x,y)\\) denotes a transport plan that moves the amount of dirt from \\(x\\) to \\(y\\), and \\(W_1(\mu,\nu)\\) is the minimum cost of turning \\(\mu\\) into \\(\nu\\). 

## Formulating 1st Wasserstein Distance as a Dual Problem

We can remove the constraint in Equation \\((1)\\) by adding an additional term

$$
\begin{align}
\mathcal L(\gamma,f, g)=\inf_\gamma\int_{M\times M}d(x,y)d\gamma(x,y)+\underbrace{\sup_{f,g}\int fd\mu(x)+\int gd\nu(y)-\int(f(x)+g(y))d\gamma(x,y)}_{=\begin{cases}0&\text{if }\gamma\in\Gamma(\mu,\nu)\\\\infty&\text{otherwise}\end{cases}}
\end{align}
$$

The supremum term impose the constraint \\(\gamma\in\Gamma(\mu,\nu)\\): it's \\(0\\) when the constraint is satisfied, and otherwise we can choose suitable \\(f\\) and \\(g\\) to make it infinity. 

Now we move the \\(\sup_{f,g}\\) outside because the first term does not depend on \\(f\\) and \\(g\\)

$$
\begin{align}
\mathcal L(\gamma,f, g)=\inf_\gamma\sup_{f,g}\int_{M\times M}d(x,y)d\gamma(x,y)+\int fd\mu(x)+\int gd\nu(y)-\int(f(x)+g(y))d\gamma(x,y)
\end{align}
$$

To get the dual problem, we swap \\(\inf\\) and \\(\sup\\)

$$
\begin{align}
\mathcal L(\gamma,f, g)=\sup_{f,g}\inf_\gamma\int_{M\times M}d(x,y)-(f(x)+g(y))d\gamma(x,y)+\int fd\mu(x)+\int gd\nu(y)
\end{align}
$$

from which we obtain the dual problem as

$$
\begin{align}
\sup_{f(x)+g(y)\le d(x,y)}\int fd\mu(x)+\int gd\nu(y)\tag 2
\end{align}
$$


In mathematics, Equation \\((1)\\) is known as the Monge problem and Equation \\((2)\\) is usually called Kantorovich duality. 

Note that because Equation \\((1)\\) is linear w.r.t. \\(\gamma\\), Equation \\((2)\\) is linear w.r.t. \\(f\\) and \\(g\\), strong duality holds and Equation \\((1)\\) is equal to Equation \\((2)\\) at the optimum.

## Applying the Lipschitz constraint

Let's assume we have the function \\(f\\). Because \\(d\mu, d\nu\ge 0\\), the supremum is acquired when \\(g\\) is at its largest, which is obtained, according to the constraint, at \\(g(y)=\inf_x d(x,y) - f(x)\\). This function is often called \\(c\\)-transform([Cuturi](#ref1) call this \\(D\\) transform but I found \\(c\\)-transform is more well-known in literature) and denoted by \\(f^c(y)=g(y)=\inf_x d(x,y) - f(x)\\). Replacing \\(g\\) with \\(f^c\\), we rewrite Equation \\((2)\\) as

$$
\begin{align}
\sup_f\int fd\mu(x)+\int f^cd\nu(y)\tag {3}
\end{align}
$$

An interesting property of \\(f^c\\) is that when \\(f\\) is 1-Lipschitz, \\(f^c\\) is Lipschitz too as \\(d(x, y)=\vert x-y\vert \\) is 1-Lipschitz. For all \\(x\\) and \\(y\\), \\(f^c\\) being 1-Lipchitz gives us

$$
\begin{align}
&&|f^c(y)-f^c(x)|\le&|y-x|\\\
&&&\color{red}{\text{without loss of generality, we assume }f^c(y)\ge f^c(x)}\\\
\Longrightarrow&&-f^c(x)\le& |y-x|-f^c(y)\\\
\Longrightarrow&&-f^c(x)\le& \inf_y|y-x|-f^c(y)\le -f^c(x)
\end{align}
$$

where the last inequality holds by choosing \\(y=x\\) in the infimum. This gives us \\(-f^c(x)=\inf_y\vert y-x\vert -f^c(y)\\). Also, noticing that \\(f^c(y)=g(y)\\) and \\(f(x)=\inf_y\vert y-x\vert -g(y)\\)(obtained through the same \\(c\\)-transform), we get \\(f(x)=-f^c(x)\\). Substituting \\(f^c(x)=-f(x)\\) in Equation \\((3)\\), we get

$$
\begin{align}
\sup_{f\text{ is Lipschitz}}\int f(d\mu-d\nu)
\end{align}
$$

which is the dual form of 1-Wasserstein distance discussed in the WGAN paper.

## Why Wasserstein is preferred over KL and JS distance?

The simple answer is that Wasserstein distance is smoother than KL and JS distance. Consider two distributions, \\(P\\) and \\(Q\\), 

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

When \\(\theta\ne 0\\), we have

$$
\begin{align}
D_{KL}(P,Q)&=+\infty\\\
D_{KL}(Q,P)&=+\infty\\\
D_{JS}(P,Q)&={1\over 2}\left(\int_{x=0,y\in[0,1]}1\cdot\log{1\over {.5}}dxdy
+\int_{x=\theta,y\in[0,1]}1\cdot\log{1\over {.5}}dxdy\right) = \log 2\\\
W_1(P,Q)&=|\theta|
\end{align}
$$

When \\(\theta=0\\), the two distributions are overlapped and all of the above metrics are zero. We can see that the KL divergence blows up to infinity when two distributions are disjoint, while the JS divergence stays constant when two distribution are disjoint and suddenly drop to zero once they overlapped. Only the Wasserstein is continuous and smooth.

## References

<a name="ref1"></a>Marco Cuturi - A Primer on Optimal Transport Part 2: https://www.youtube.com/watch?v=1ZiP_7kmIoc

<a name="ref2"></a>Vincent Herrmann's blog: https://vincentherrmann.github.io/blog/wasserstein/

Arjovsky, Martin, Soumith Chintala, and L´eon Bottou. 2017. “Wasserstein GAN.”

## Supplementary

### Another Way of Deriving Kantorovich Duality

We can define the 1st Wasserstein distance as a primal problem:

$$
\begin{align}
&\text{minimize}&\pmb c^\top\pmb x\\\
&s.t.&\pmb A\pmb x=\pmb b\\\
&&\pmb x\ge 0
\end{align}
$$

where \\(\pmb x=vec(\Gamma)\\) is a flatten vector of \\(\Gamma\\), \\(\pmb c=vec(D)\\) is a flatten vector of the set of distance \\(D=\{d(x,y)\vert x,y\in M\}\\), \\(\pmb b=\begin{bmatrix}\pmb \mu\\\\pmb \nu\end{bmatrix}\\) is the concatenation of marginals \\(\pmb \mu\\) and \\(\pmb \nu\\)(we use the bold font to denote values of the corresponding functions), and \\(\pmb A\\) is a sparse binary matrix such that \\(\pmb A[i]\pmb x=\pmb b[i]\\), which enforces the marginal constraints \\(\int\gamma(x,y)dy=\mu(x)\\) and \\(\int\gamma(x,y)dx=\nu(y)\\). It may be clearer to divide \\(\pmb A\\) into two matrices, where we have \\(\pmb A_1\pmb x=\pmb \mu\\) and \\(\pmb A_2\pmb x=\pmb \nu\\). For \\(\pmb A_1\\) and \\(\pmb A_2\\), each column corresponds to a transfer plan in \\(\pmb x\\) and there is exactly one \\(1\\) in each column as transport plans are only counted once. For example, \\(\pmb x[k]\\) is a tranport plan from \\(x\\) to \\(y\\), then \\(\pmb A_1[x,k]=1\\) and \\(\pmb A_2[y,k]=1\\). Therefore, there are two \\(1\\)s in each column of \\(\pmb A\\). This property will come in handy next when we discuss the dual.

The dual is a maximization problem defined as

$$
\begin{align}
&\text{maximize}&\pmb b^\top\pmb y\\\
&s.t.&\pmb A^\top\pmb y\le\pmb c
\end{align}
$$

Notice that the sign constraints on \\(\pmb y\\) is removed as the primal constraints becomes equality(see the proof in Supplementary Materials for details).

Like \\(\pmb b=\begin{bmatrix}\pmb \mu\\\\pmb \nu\end{bmatrix}\\), let \\(\pmb y=\begin{bmatrix}\pmb f\\\\pmb g\end{bmatrix}\\). Following our previous analysis on \\(\pmb A\\), we can see that, for each constraint \\(\pmb c[k]=d(x,y)\\), we have \\(\pmb A_1^\top[k,x]\pmb f[x] + \pmb A_2^\top[k,y]\pmb g[y]=\pmb f[x]+\pmb g[y]\le d(x,y)\\). Now we rewrite the dual as 

$$
\begin{align}
&\text{maximize}&\pmb \mu^\top \pmb f+\pmb \nu^\top \pmb g&\\\
&s.t.&\pmb f[x]+\pmb g[y]\le d(x,y)&\quad \forall x,y
\end{align}
$$


### Proof of Strong Duality

Following the similar process in the [previous post]({{ site.baseurl }}{% post_url 2020-04-07-dual %}), let 

$$
\begin{align}
\pmb{\hat A}=\begin{bmatrix}-\pmb A\\\\pmb c^\top\end{bmatrix},
\pmb{\hat b}=\begin{bmatrix}-\pmb b\\\\tau-\epsilon\end{bmatrix}
\end{align}
$$

where \\(\tau=\pmb c^\top\pmb x^\*\\) is the optimal value of the dual, \\(\epsilon\ge 0\\) is an arbitrary value. Because \\(\tau\\) is the optimal value, there is no feasible \\(\pmb x\\) such that \\(\pmb c^\top\pmb x=\tau-\epsilon\\). Therefore, there is no \\(\pmb x\in\mathbb R^n\\) such that

$$
\begin{align}
\begin{bmatrix}-\pmb A\\\\pmb c^\top\end{bmatrix}\pmb x
\le\begin{bmatrix}-\pmb b\\\\tau-\epsilon\end{bmatrix}
\end{align}
$$

By the variant of the Farkas' Lemma, there exists \\(\pmb{\hat y}=\begin{bmatrix}\pmb y\\\ \alpha\end{bmatrix}\ge\pmb 0 \\) such that

$$
\begin{align}
\begin{bmatrix}-\pmb A^\top&\pmb c\end{bmatrix}\begin{bmatrix}\pmb y\\\\alpha\end{bmatrix}= \pmb 0,
\quad \begin{bmatrix}-\pmb b^\top&\tau-\epsilon\end{bmatrix}\begin{bmatrix}\pmb y\\\\alpha\end{bmatrix} <0\\\
\end{align}
$$

Thus, we have

$$
\begin{align}
\pmb A^\top\pmb y=\alpha\pmb c,\quad \pmb b^\top\pmb y >\alpha(\tau-\epsilon)
\end{align}
$$

If \\(\alpha=0\\), then the primal is infeasible or unbounded—for any feasible solution to the dual \\(\pmb b^\top\pmb y_\*\\), we can always find a larger feasible solution \\(\pmb b^\top(\pmb y_\*+\pmb y)\\) because \\(\pmb A^\top\pmb y= \pmb 0\\) and \\(\pmb b^\top\pmb y>0\\). If \\(\alpha>0\\), without loss of generality, by scaling \\(\pmb {\hat y}\\) we may assume that \\(\alpha=1\\). This gives us \\(\pmb b^\top\pmb y>\tau-\epsilon\\). By the weak dual theorem, we have \\(\tau= \pmb c^\top \pmb x\ge\pmb b^\top\pmb y\ge\tau-\epsilon\\). Since \\(\epsilon\\) is arbitrary, we have \\(\pmb c^\top \pmb x=\pmb b^\top\pmb y\\).

