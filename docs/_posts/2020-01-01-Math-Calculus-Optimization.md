---
title: "Mathematics for Machine Learning — Calculus and Optimization"
excerpt: "in which we discuss calculus used in machine learning/deep learning"
categories:
  - Mathematics
---

## Fundamental Theorem of Linear Algebra

## Extrama

**Extrama** is either minima or maxima depending on the objective function.

For a set of input $$\mathcal X\subseteq \mathbb R^{d}$$, called the **feasible set**, if $$\mathcal X$$ is the entire domain of the function being optimized, we say that the problem is **unconstrained**. Otherwise, the problem is **constrained** and may be much harder to solve.

## The chain rule

**Proposition.** Suppose $$f:\mathbb R^m \rightarrow\mathbb R^k$$ and $$g:\mathbb R^n \rightarrow\mathbb R^m$$. Then $$f\circ g:\mathbb R^n \rightarrow \mathbb R^k $$ and $$\pmb J_{f\circ g}(\pmb x)=\pmb J_f(g(\pmb x))\pmb J_g(\pmb x)$$

We can easily obtain the gradient form of the chain rule $$\nabla(f\circ g)(\pmb x)=\nabla g(\pmb x)\nabla f(g(\pmb x))$$

### Taylor's theorem

**Theorem.** (Taylor's theorem) Suppose $$f:\mathbb R^d\rightarrow \mathbb R$$ is continuously differentiable, and let $$\Delta\pmb x=\in\mathbb R^d$$, Then there exists $$t\in(0,1)$$ such that

$$
\begin{align}
f(\pmb x+\Delta\pmb x)=f(\pmb x)+\nabla f(\pmb x+t\Delta\pmb x)^\top\Delta\pmb x
\end{align}
$$

Furthermore, if $$f$$ is twice continuously, then

$$
\begin{align}
\nabla f(\pmb x+\Delta\pmb x)=\nabla f(\pmb x)+\int_0^1\nabla^2f(\pmb x+t \Delta\pmb x)\Delta\pmb xdt
\end{align}
$$

And there exists $$t\in(0,1)$$ such that

$$
\begin{align}
f(\pmb x+\Delta\pmb x)=f(\pmb x)+\nabla f(\pmb x)^\top\Delta\pmb x+{1\over 2}\Delta\pmb x^\top\nabla^2f(\pmb x+t\Delta\pmb x)\Delta\pmb x
\end{align}
$$


More often than not, we approximate a first- or second-order Taylor's theorem

$$
\begin{align}
f(\pmb x+\Delta\pmb x)\approx&f(\pmb x)+\nabla f(\pmb x)^\top\Delta\pmb x\\\
f(\pmb x+\Delta\pmb x)\approx&f(\pmb x)+\nabla f(\pmb x)^\top\Delta\pmb x+{1\over 2}\Delta\pmb x^\top\nabla^2f(\pmb x)\Delta\pmb x
\end{align}
$$


## The Jacobian

The **Jacobian** of $$f:\mathbb R^n\rightarrow\mathbb R^m$$ is a $$m \times n$$ matrix of first order partial derivatives

$$
\begin{align}
\mathbf J_f=\begin{bmatrix}
{\partial f_1\over\partial x_1}&\dots&{\partial f_n\over\partial x_n}\\\
\vdots&\ddots&\vdots\\\
{\partial f_m\over\partial x_1}&\dots&{\partial f_m\over\partial x_n}
\end{bmatrix}
\end{align}
$$

Note the special case $$m=1$$, the gradient $$\nabla f=\mathbf J_f^\top$$

## The Hessian

<figure>
  <img src="{{ '/images/math/DL-Figure4.4.png' | absolute_url }}" alt="" width="1000">
  <figcaption>Source: Goodfellow et al. Deep Learning</figcaption>
  <style>
    figure figcaption {
    text-align: center;
    }
  </style>
</figure>

The **second derivative** is a derivative of a derivative. As the first derivative reflects how the objective will change as we vary the input, the second derivative tells us how the first derivative will change as we vary the input. This is important because it tells us whether a gradient step will cause as much of an improvement as we would expect based on the gradient alone. We can think of the second derivative as measuring **curvature** . Suppose we have a quadratic function (many functions that arise in practice are not quadratic but can be approximated well as quadratic, at least locally). If such a function has a second derivative of zero, then there is no curvature. It is a perfectly ﬂat line, and its value can be predicted using only the gradient. If the gradient is $$1$$, then we can make a step of size $$\epsilon$$ along the negative gradient, and the cost function will decrease by $$\epsilon$$. If the second derivative is negative, the function curves downward, so the cost function will actually decrease by more than $$\epsilon$$. Finally, if the second derivative is positive, the function curves upward, so the cost function can decrease by less than $$\epsilon$$. Figure 4.4 shows an example of how diﬀerent forms of curvature aﬀect the relationship between the value of the cost function predicted by the gradient and the true value.

The **Hessian** matrix of $$f:\mathbb R^d\rightarrow \mathbb R$$ is a matrix of the second derivatives

$$
\begin{align}
\pmb H=\nabla^2 f=\begin{bmatrix}
{\partial^2 f\over\partial x_1^2}&\dots&{\partial^2 f\over\partial x_1\partial x_d}\\\
\vdots&\ddots&\vdots\\\
{\partial^2 f\over\partial x_d\partial x_1}&\dots&{\partial^2 f\over\partial x_d^2}
\end{bmatrix}
\end{align}
$$

If the second partial derivatives are continuous, the order of differentiation can be interchanged (Clairaut's theorem), so the Hessian matrix will be symmetric.

The Hessian is used in some optimization algorithms such as Newton's method. It's *expensive to calculate* but can drastically reduce the number of iterations needed to converge to a local optimum by providing information about the curvature of $$f$$.

We often say $$\pmb x$$ is a **critical point** if $$\nabla f(\pmb x)=\pmb 0$$ or no partial derivative exists at $$\pmb x$$. We can use the Hessian matrix to determine if a critical point is local minimum or a local maximum. Specifically if the Hessian at $$\pmb x$$ is positive definite, $$\pmb x$$ is a local minimum. When $$\pmb H$$ is symmetric, the following statements are equivalent

1. $$\pmb H$$ is positive definite
2. $$\pmb x^\top\pmb H\pmb x>0$$ for any nonzero real vector $$\pmb x$$
3. The eigenvalues of $$\pmb H$$ are all positive (by the Rayleigh quotient)
4. The determinant is positive: $$\vert \pmb H\vert >0$$ ($$\vert \pmb H\vert =\prod_i\lambda_i$$)
5. The diagonal entries of $$\pmb H$$ are positive (conceiving the case for $$i\in d$$, $$\pmb e_i^\top\pmb H\pmb e_i>0$$, where $$\pmb e_i$$ is a vector of zeros except for $$1$$ in the $$i^{th}$$ place)

If the Hessian at $$\pmb x$$ is negative definite, $$\pmb x$$ is a local maximum. The following statements are equivalent

1. $$\pmb H$$ is negative definite
2. $$\pmb x^\top\pmb H\pmb x<0$$ for any nonzero real vector $$\pmb x$$
3. The eigenvalues of $$\pmb H$$ are all negative (by the Rayleigh quotient)
4. The determinant is positive when $$d$$ is even and negative when $$d$$ is odd: $$\vert \pmb H\vert >0$$ ($$\vert \pmb H\vert =\prod_i\lambda_i$$)
5. The diagonal entries of $$\pmb H$$ are negative (conceiving the case for $$i\in d$$, $$\pmb e_i^\top\pmb H\pmb e_i<0$$, where $$\pmb e_i$$ is a vector of zeros except for $$1$$ in the $$i^{th}$$ place)

### Conditions for local minima

We prove the Hessian is positive definite for local minimum from the Taylor's theorem. 

$$
\begin{align}
f(\pmb x^*+\Delta\pmb x)=f(\pmb x^*)+\nabla f(\pmb x^*)^\top\Delta\pmb x+{1\over 2}\Delta\pmb x^\top\nabla^2f(\pmb x^*+t\Delta\pmb x)\Delta\pmb x\ge f(\pmb x^*)
\end{align}
$$

where the last inequality holds when the gradient $$\nabla f(\pmb x^*)=\pmb 0$$ and the Hessian $$\pmb H(\pmb x^*)=\nabla^2 f(\pmb x^*+t\Delta\pmb x)$$ is positive semi-definite. 

Note that $$\nabla f(\pmb x^*)=\pmb 0$$ and $$\pmb H(\pmb x^*)$$ is *positive semi-definite* is a necessary but not a sufficient condition for $$\pmb x^*$$ being a local minimum. For example, $$f(x)=x^3$$ has $$f'(0)=0$$ and $$f''(0)=0$$ but $$f$$ has a saddle point at $$x=0$$. The function $$f(x)=-x^4$$ is an even worse offender — it has the same gradient and Hessian at $$x=0$$ but $$x$$ is a strict local maximum for this function.

On the other hand, $$\nabla f(\pmb x^*)=0$$ and $$\pmb H(\pmb x^*)$$ is *positive definite* is a sufficient but not a necessary condition for $$\pmb x^*$$ being a local minimum. For example $$f(x)=x^4$$ is a local minimum at $$0$$ but it has zero gradient and Hessian at $$x=0$$.

### Directional second derivative

The second derivative in a specific direction represented by a unit $$\pmb d$$ is given by $$\pmb d^\top\pmb H\pmb d$$. When $$\pmb d$$ is an eigenvector of $$\pmb H$$, the second derivative in that direction is given by the corresponding eigenvalue. For other directions of $$\pmb d$$, the directional second derivative is a weighted average of all of the eigenvalues, with weights between $$0$$ and $$1$$, and eigenvectors that have smaller angle with $$\pmb d$$ receiving more weight. 

The directional second derivative tells us how well we can expect a gradient descent to perform. Make a second-order Taylor series approximation

$$
\begin{align}
f(\pmb x+\Delta\pmb x)\approx f(\pmb x)+\pmb g^\top\Delta\pmb x+{1\over 2}\Delta\pmb x^\top\pmb H\Delta\pmb x
\end{align}
$$

where $$\pmb g$$ is the gradient and $$\pmb H$$ is the Hessian at $$\pmb x$$. If we use $$\Delta\pmb x=-\epsilon \pmb g$$, we obtain

$$
\begin{align}
f(\pmb x-\epsilon\pmb g)\approx f(\pmb x)-\epsilon\pmb g^\top\pmb g+{1\over 2}\epsilon^2\pmb g^\top\pmb H\pmb g
\end{align}
$$

There are three terms here: the original value of the function, the expected improvement due to the slope of the function, and the correction we must apply to account for the curvature of the function. When $$\pmb g^\top\pmb H\pmb g$$ is zero or negative, the Taylor series approximation predicts that increasing $$\epsilon$$ will always decrease $$f$$ though this is not true in practice as the Taylor series approximation becomes inaccurate for large $$\epsilon$$. When $$\pmb g^\top\pmb H\pmb g$$ is positive, we can compute the optimal step size using the partial derivative of the Taylor series approximation w.r.t. $$\epsilon$$, which gives 

$$
\begin{align}
\epsilon^*={\pmb g^\top\pmb g\over\pmb g^\top\pmb H\pmb g}
\end{align}
$$

In the worst case, when $$\pmb g$$ aligns with the eigenvector of $$\pmb H$$ corresponding to the maximum eigenvalue $$\lambda_\max$$, the optimal step size is given by $${1\over\lambda_\max}$$. To the extent that the function we minimize can be approximated well by a quadratic function, the eigenvalues of Hessian thus determine the scale of the learning rate.

### Inefficiency of gradient descent

In multiple dimensions, there is a diﬀerent second derivative for each direction at a single point. The condition number of the Hessian at this point measures how much the second derivatives diﬀer from each other. When the Hessian has a poor condition number, gradient descent performs poorly. This is because in one direction, the derivative increases rapidly, while in another direction, it increases slowly. Gradient descent is unaware of this change in the derivative so it does not know that it needs to explore preferentially in the direction where the derivative remains negative for longer. It also makes it diﬃcult to choose a good step size. The step size must be small enough to avoid overshooting the minimum and going uphill in directions with strong positive curvature. This usually means that the step size is too small to make signiﬁcant progress in other directions with less curvature.

### Newton's method

**Newton's method** finds the root $$\pmb x$$ such that $$f(\pmb x)\approx f(\pmb x_n)+\nabla f(\pmb x_n)^\top(\pmb x-\pmb x_n)=0$$ by iteratively computing $$\pmb x_{n+1}=\pmb x_n-{(\nabla f(\pmb x_n)^\top)^{-1}f(\pmb x_n)}$$ starting from some random guess $$\pmb x_0$$. It's only suitable for the case where the initial guess is near a root.

For an optimization problem $$f(\pmb x)$$, we can apply Newton's method to solve $$\nabla f(\pmb x)=0$$ and then check the Hessian to determine if the solution is a extrama we expected. The iterative rule in that case becomes $$\pmb x_{n+1}=\pmb x_n-\pmb H^{-1}(\pmb x_n)\nabla f(\pmb x_n)$$. Newton's method takes advantage the information in the Hessian, and therefore it usually converges faster than gradient descent. On the other hand, depending on the initial guess, it's easily attracted a saddle point or an opposite extrama. Therefore, a sanity check is usually required after the convergence.

## Convex

### Convex set

<figure>
  <img src="{{ '/images/math/Math-for-ML-Figure1.png' | absolute_url }}" alt="" width="1000">
  <figcaption></figcaption>
  <style>
    figure figcaption {
    text-align: center;
    }
  </style>
</figure>

A set $$\mathcal X\in\mathbb R^d$$ is **convex** if 

$$
\begin{align}
t\pmb x+(1-t)\pmb y\in\mathcal X
\end{align}
$$

for all $$\pmb x,\pmb y\in \mathcal X$$ and all $$t\in[0,1]$$.

Geometrically, this means that all the points on the line segment between any two points in $$\mathcal X$$ are also in $$\mathcal X$$. See Figure 1 for an visual.

### Basics of convex functions

A function $$f$$ is **convex** if

$$
\begin{align}
f(t\pmb x+(1-t)\pmb y)\le tf(\pmb x)+(1-t)f(\pmb y)
\end{align}
$$

for all $$\pmb x,\pmb y\in \mathcal X$$ and all $$t\in[0,1]$$.

**Proposition.** Let $$\mathcal X$$ be a convex set. If $$f$$ is convex, then any local minimum of $$f$$ in $$\mathcal X$$ is also a global minimum.

**Proposition. ** Let $$\mathcal X$$ be a convex set. If $$f$$ is strictly convex, then there exists at most one local minimum, which is also the unique global minimum.

### Showing that a function is convex

**Proposition.** Norms are convex.

**Proof.** For all $$\pmb x,\pmb y\in V$$ and $$t\in[0,1]$$

$$
\begin{align}
\Vert t\pmb x+(1-t)\pmb y\Vert\le \Vert t\pmb x\Vert+\Vert(1-t)\pmb y\Vert=t\Vert\pmb x\Vert+(1-t)\Vert\pmb y\Vert
\end{align}
$$

**Proposition.** Suppose $$f$$ is differentiable. Then $$f$$ is convex if and only if

$$
\begin{align}
f(\pmb x)\ge f(\pmb y)+\langle\nabla f(\pmb y),\pmb x-\pmb y\rangle
\end{align}
$$

for all $$\pmb x,\pmb y\in\text{dom} f$$.

**Proof.** 

$$
\begin{align}
&f(t\pmb x+(1-t)\pmb y)\le tf(\pmb x)+(1-t)f(\pmb y)=f(\pmb y)+t(f(\pmb x)-f(\pmb y))\\\
\Longleftrightarrow\quad&{f(t(\pmb x-\pmb y)+\pmb y)- f(\pmb y)\over t}\le f(\pmb x)-f(\pmb y)\\\
\Longleftrightarrow\quad&{f(\pmb y)+t\nabla f(\pmb y)^\top(\pmb x-\pmb y)-f(\pmb y)\over t}\le f(\pmb x)-f(\pmb y)\\\
\Longleftrightarrow\quad&\langle\nabla f(\pmb y),\pmb x-\pmb y\rangle+f(\pmb y)\le f(\pmb x)
\end{align}
$$

for all $$\pmb x,\pmb y\in \mathcal X$$ and all $$t\in[0,1]$$.

**Proposition** Suppose $$f$$ is twice differentiable. Then

1. $$f$$ is convex if and only if $$\nabla^2f(\pmb x)$$ is *positive semi-definite* for all $$\pmb x\in\text{dom} f$$.

2. If $$\nabla^2f(\pmb x)$$ is *positive definite* for all $$\pmb x\in \text{dom} f$$, then $$f$$ is strictly convex

**Proof.** We prove (1) and (2) immediately follows

($$\Longrightarrow$$)Define $$g(t)=f(t\pmb x+(1-t)\pmb y)$$, for all $$\pmb x,\pmb y\in \mathcal X$$ and $$t\in[0,1]$$. We have 

$$
\begin{align}
g'(t)=&(\pmb x-\pmb y)^\top\nabla f(t\pmb x+(1-t)\pmb y)\\\
g''(t)=&(\pmb x-\pmb y)^\top\nabla^2 f(t\pmb x+(1-t)\pmb y)(\pmb x-\pmb y)
\end{align}
$$

By the Taylor theorem, we have

$$
\begin{align}
g(0)\ge& g(t)-tg'(t) &\text{with }x=t,&\Delta x=-t\\\
g(1)\ge& g(t)+(1-t)g'(t) &\text{with }x=t,&\Delta x=1-t
\end{align}
$$

for $$\nabla^2 f(\pmb x)$$ being positive semi-definite. Combining them, we obtain

$$
\begin{align}
&(1-t)g(0)+tg(1)\ge g(t)\\\
\Longleftrightarrow\quad&(1-t)f(\pmb y) + t f(\pmb x)\ge f(t\pmb x+(1-t)\pmb y)
\end{align}
$$

 Therefore, $$f(\pmb x)$$ is convex.

($$\Longleftarrow$$)As before, defining $$g(t)=f(t\pmb x+(1-t)\pmb y)$$, we have $$g(0)=f(\pmb y)$$ and $$g(1)=f(\pmb x)$$. For a convex function $$f$$, we have

$$
\begin{align}
(1-t)g(0)+tg(1)\ge& g(t)\\\
(1-t)\Big(g(t)-tg'(t)+{1\over 2}t^2g''(t)\Big)+t\Big(g(t)+(1-t)g'(t)+{1\over 2}(1-t)^2g''(t)\Big)\ge&g(t)\\\
g(t)+{1\over 2}\Big((1-t)t^2+t(1-t)^2\Big)g''(t)\ge& g(t)\\\
g''(t)\ge&0
\end{align}
$$

where the last step is obtained because of $$t\in[0,1]$$. Therefore, $$\nabla^2 f(\pmb x)$$ is positive semi-definite.

**Definition.** $$f$$ is **$$m$$-strongly convex** if 

$$
\begin{align}
f(t\pmb x+(1-t)\pmb y)\le tf(\pmb x)+(1-t)f(\pmb y)-{1\over 2}mt(1-t)\Vert\pmb x-\pmb y\Vert^2
\end{align}
$$

If $$f$$ is twice differentiable, then it is $$m$$-strongly convex if and only if $$\nabla^2 f(\pmb x)-mI$$ is positive semi-definite for all $$\pmb x\in\text{dom}f$$. This equivalent to requiring that the minimum eigenvalue of $$f$$ be at least $$m$$ for all $$\pmb x$$.

**Proposition.** If $$f$$ is convex and $$\alpha\ge 0$$, then $$\alpha f$$ is convex

**Proposition.** If $$f$$ and $$g$$ are convex, then $$f+g$$ is convex. Furthermore, if $$g$$ is strictly convex, then $$f+g$$ is strictly convex, and if $$g$$ is $$m$$-strongly convex, then $$f+g$$ is $$m$$-strongly convex.

**Proof.** We only prove the case where $$g$$ is $$m$$-strongly convex, the other two can be obtain in a similar vein. By definition, we have

$$
\begin{align}
f(t\pmb x+(1-t)\pmb y)\le& tf(\pmb x)+(1-t)f(\pmb y)\\\
g(t\pmb x+(1-t)\pmb y)\le& tg(\pmb x)+(1-t)g(\pmb y)-{1\over 2}mt(1-t)\Vert\pmb x-\pmb y\Vert^2
\end{align}
$$

Adding them up, we obtain

$$
\begin{align}
(f+g)(t\pmb x+(1-t)\pmb y)\le t(f+g)(\pmb x)+(1-t)(f+g)(\pmb y)-{1\over 2}mt(1-t)\Vert\pmb x-\pmb y\Vert^2
\end{align}
$$

So $$f+g$$ is $$m$$-strongly convex

**Proposition.** If $$f$$ is convex, then $$g(\pmb x)=f(\pmb A\pmb x+\pmb b)$$ is convex for any appropriately-sized $$\pmb A$$ and $$\pmb b$$.

**Proof.**

$$
\begin{align}
g(t\pmb x+(1-t)\pmb y)=&f(\pmb A(t\pmb x+(1-t)\pmb y)+\pmb b)\\\
=&f(t(\pmb A\pmb x+\pmb b)+(1-t)(\pmb A\pmb y+\pmb b))\\\
\le &tf(\pmb A\pmb x+\pmb b)+(1-t)f(\pmb A\pmb y+\pmb b)\\\
=&tg(\pmb x)+(1-t)g(\pmb y)
\end{align}
$$

Thus $$g$$ is convex.

**Proposition.** if $$f$$ and $$g$$ are convex, then $$h(\pmb x)=\max\{f(\pmb x),g(\pmb y)\}$$ is convex

**Proof.**

$$
\begin{align}
h(t\pmb x+(1-t)\pmb y)\le &\max\{tf(\pmb x)+(1-t)f(\pmb y),tg(\pmb x)+(1-t)g(\pmb y)\}\\\
\le&\max\{tf(\pmb x),tg(\pmb y)\} +\max\{(1-t)f(\pmb y),(1-t)g(\pmb y)\}\\\
=&t\max\{f(\pmb x),g(\pmb x)\} +(1-t)\max\{f(\pmb y),g(\pmb y)\}\\\
=&th(\pmb x)+(1-t)h(\pmb y)
\end{align}
$$


### Examples

Functions that are convex but not strictly convex:

- $$f(\pmb x)=\pmb w^\top\pmb x+\alpha$$

- $$f(\pmb x)=\Vert\pmb x\Vert_1$$

Functions that are strictly but not strongly convex:

- $$f(x)=x^4$$
- $$f(x)=\exp(x)$$
- $$f(x)=-\log x$$

Functions that are strongly convex:

- $$f(\pmb x)=\Vert\pmb x\Vert_2^2$$

## References

Thomas, Garrett. 2018. “Mathematics for Machine Learning” 56 (5): 1–47.

Ian, Goodfellow, Bengio Yoshua, and Courville Aaron. 2016. *Deep Learning*. MIT Press.