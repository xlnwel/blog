---
title: "Mathematics for Machine Learning — Numerical Computation"
excerpt: "in which we we discuss probabilities used in machine learning/deep learning"
categories:
  - Mathematics

---

# Numerical Computation

## Gradient-based optimization

We define the **directional derivative** in direction $$\pmb u$$(a unit vector) as the slope of the function $$f$$ in direction $$\pmb u$$. In other word, the directional derivative is $${\partial\over\partial\alpha}f(\pmb x+\alpha\pmb u)$$, the derivative of the function $$f(\pmb x+\alpha\pmb u)$$ with respect to $$\alpha$$, evaluated at $$\alpha =0$$. To minimize $$f$$, we would like to find the direction in which $$f$$ decreases the fastest, which is equal to find $$\pmb u$$ that minimizes the directional derivative:

$$
\begin{align}
&\min_{\pmb u,\pmb u^\top\pmb u=1}{\partial\over\partial\alpha}f(\pmb x+\alpha\pmb u)\\\
=&\min_{\pmb u,\pmb u^\top\pmb u=1}\pmb u^\top\nabla_{\pmb x}f(\pmb x)&\color{red}{\text{the chain rule, evaluated at } \alpha=0}\\\
=&\min_{\pmb u,\pmb u^\top\pmb u=1}\underbrace{\Vert\pmb u\Vert_2}_{=1}\underbrace{\Vert\nabla_{\pmb x}f(\pmb x)\Vert_2}_{\text{irrelavent to }\pmb u}\cos\theta\\\
=&\min_{\pmb u}\cos\theta
\end{align}
$$

Since $$\min_{\pmb u}\cos\theta$$ is minimized when $$\pmb u$$ points to the opposite direction as the gradient. We can decrease $$f$$ by moving in the direction of the negative gradient. This is known as the **method of steepest descent** or **gradient descent**.

Gradient descent proposes a new point

$$
\begin{align}
\pmb x'=\pmb x-\epsilon\nabla_{\pmb x}f(\pmb x)
\end{align}
$$

where $$\epsilon$$ is the *learning rate*. 

**Line search** evaluates $$f(\pmb x-\epsilon\nabla_{\pmb x}f(\pmb x))$$ for several values of $$\epsilon$$ and choose the one that results in the smallest objective function value.

## Constrained Optimization

Consider a constraint optimization pdefined as bellow

$$
\begin{align}
\min_{\pmb x}\quad &f(\pmb x)&\\\
s.t.\quad
&g^{(i)}(\pmb x)=0 \quad&\text{for }&i=1,...,n\\\
&h^{(j)}(\pmb x)\le0 \quad&\text{for }&j=1,...,m\\\
\end{align}
$$

The **Karush-Kuhn-Tucker** (KKT) approach provides a general solution for such problems. With the KKT approach, we introduce the **generalized Lagrangian** or **generalized Lagrange function** as

$$
\begin{align}
\mathcal L(\pmb x,\pmb \lambda,\pmb \alpha)=f(\pmb x)+\sum_i\lambda_ig^{(i)}(\pmb x)+\sum_j\alpha_jh^{(j)}(\pmb x)
\end{align}
$$

where $$\lambda_i$$ and $$\alpha_j$$ are referred to as the KKT multipliers. The generalized Lagrangian differs from the Lagrangian in that it allows equality constraints other than inequality constraints. We can solve the generalized Lagrangian by minimizing $$\pmb x$$ and maximizes the KKT multipliers:

$$
\begin{align}
\min_{\pmb x}\max_{\pmb \lambda}\max_{\pmb \alpha\ge \pmb 0}\mathcal L(\pmb x,\pmb \lambda,\pmb \alpha)
\end{align}
$$

This has the same optimal points as the original constraint optimization problem because when the constraints are satisfied, the two terms related to the KKT multipliers vanishes. In particular, we will have $$\alpha_j=0$$ for $$h^{(j)}(\pmb x)<0$$ and $$\alpha_j>0$$ for $$h^{(j)}(\pmb x)=0$$. We say a constraint $$h^{(j)}(\pmb x)$$ is active if $$h^{(j)}(\pmb x)=0$$. If a constraint is not active, then the solution to the problem found using that constraint would remain at least a local solution if that constraint were removed, which might not be true for an active constraint.

There are several necessary conditions, but not always sufficient conditions, for a point to be optimal:

- The gradient of the generalized Lagrangian is zero
- All constraints on both $$\pmb x$$ and the KKT multipliers are satisfied
- The inequality constraints exhibit "complementary slackness": $$\pmb \alpha\odot\pmb h(\pmb x)=\pmb 0$$

## References

Ian, Goodfellow, Bengio Yoshua, and Courville Aaron. 2016. *Deep Learning*. MIT Press.