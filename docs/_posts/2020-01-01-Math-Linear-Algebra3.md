---
title: "Mathematics for Machine Learning — Linear Algebra Part 3"
excerpt: "in which we discuss linear algebra used in machine learning/deep learning"
categories:
  - Mathematics
---

# Linear Algebra Part 3

## Fundamental theorem of linear algebra

**Theorem** If $$\pmb A\in \mathbb R^{m \times n}$$, then

1. $$\text{null}(\pmb A)=\text{range}(\pmb A^\top)^\perp$$

2. $$\text{null}(\pmb A)+\text{range}(\pmb A^\top)=\mathbb R^n$$

3. Rank-nullity theorem. $$\underbrace{\dim\text{range}(\pmb A)}_{\text{rank}(\pmb A)}+\dim\text{null}(\pmb A)=n$$

4. If $$\pmb A=\pmb U\pmb \Sigma\pmb V^\top$$ is the singular decomposition of $$\pmb A$$, then the columns of $$\pmb U$$ and $$\pmb V$$ form orthogonal bases for the four "fundamental subspaces" of $$\pmb A$$

   | Subspace                    | Columns                            |
   | --------------------------- | ---------------------------------- |
   | $$\text{range}(\pmb A)$$      \vert  The first $$r$$ columns of $$\pmb U$$  |
   | $$\text{range}(\pmb A^\top)$$ \vert  The first $$r$$ columns of $$\pmb V$$  |
   | $$\text{null}(\pmb A^\top)$$  \vert  The last $$m-r$$ columns of $$\pmb U$$ |
   | $$\text{null}(\pmb A)$$       \vert  The last $$n-r$$ columns of $$\pmb V$$ |

   with $$r=\text{rank}(\pmb A)$$

**Proof.**

1. 
$$
   \begin{align}
   \text{null}(\pmb A)\Leftrightarrow&\pmb A\pmb x=\pmb 0\\\
   \Leftrightarrow& \pmb A_i\pmb x=0\quad\text{for all }i=1,\dots,m\\\
   \Leftrightarrow&(\sum_i\alpha_i\pmb A_i)\pmb x=0\\\
   \Leftrightarrow&\pmb v\pmb x=0\quad\text{for any }\pmb v\in\text{range}(\pmb A^\top)\\\
   \Leftrightarrow&\pmb x\in\text{range}(\pmb A^\top)^\perp
   \end{align}
   $$


   which proves the result

2. 1 implies that $$\text{null}(\pmb A)$$ is the orthogonal complement of $$\text{range}(\pmb A^\top)$$, so 2 holds 

3. Because $$\text{null}(\pmb A)$$ and $$\text{range}(\pmb A^\top)$$ are subspace of $$\mathbb R^n$$ and they are orthogonal complements, 3 holds

4. Following SVD, we have
   
$$
   \begin{align}
   \pmb A=&\pmb U\pmb \Sigma\pmb V^\top\\\
   =&\begin{bmatrix}
   \color{blue}{\pmb u_1}&\color{blue}{\pmb u_2}&\color{blue}{\dots}&\color{blue}{\pmb u_{r}}&\pmb u_{r+1}&\dots&\pmb u_{m}
   \end{bmatrix}
     \left[ \begin{array}{cccc|cc}
        \color{blue}\sigma_{1} & \color{blue}0 & \color{blue}\dots &  &   & \dots &  0 \\\
        \color{blue}0 & \color{blue}{\sigma_{2}}  \\\
        \color{blue}\vdots && \color{blue}\ddots \\\
          & & & \color{blue}\sigma_{r} \\\\hline
          & & & & 0 & \\\
        \vdots &&&&&\ddots \\\
        0 & & &   &   &  & 0 \\\
     \end{array} \right]
   \begin{bmatrix}
   \color{blue}{\pmb v_1^\top}\\\\color{blue}{\pmb v_2^\top}\\\\color{blue}{\vdots}\\\\color{blue}{\pmb v_r^\top}\\\\pmb v_r^\top\\\{\vdots}\\\{\pmb v_{n}^\top}
   \end{bmatrix}\\\
   =&\sum_{i=1}^{r}\sigma_i\pmb u_i\pmb v_i^\top
   \end{align}
   $$

   Look at $$\pmb A\pmb x$$
   
$$
   \pmb A\pmb x=\sum_{i=1}^r\sigma_i\pmb u_i\pmb v_i^\top\pmb x=\sum_{i=1}^r\underbrace{\sigma_i\pmb v_i^\top\pmb x}_{scalar}\pmb u_i
   $$

   Because $$\pmb A\pmb x$$ is a linear combination of $$\pmb u_{1:r}$$ and $$\pmb u_{1:r}$$ are linear independent, $$\text{range}(\pmb A)$$ is the subspace described by $$\pmb u_{1:r}$$. Following (1), we can derive $$\text{null}(\pmb A^\top)$$ is the subspace described by $$\pmb u_{r+1:m}$$. It's easy to derive the other two in a similar way.

## Operator and matrix norms

If $$T:V\rightarrow W$$ is a linear map between normed spaces $$V$$ and $$W$$, then the operator norm is defined as

$$
\begin{align}
\Vert T\Vert_{op}=\max_{\pmb x\in V\\\\pmb x\ne\pmb 0}{\Vert T\pmb x\Vert_W\over\Vert\pmb x\Vert_V}
\end{align}
$$

When the domain and codomain are $$\mathbb R^n$$ and $$\mathbb R^m$$, and the $$p$$-norm is used in both cases. Then for a matrix $$\pmb A\in\mathbb R^{m \times n}$$, we define the matrix $$p$$-norm

$$
\begin{align}
\Vert\pmb A\Vert_p=\max_{\pmb x\ne \pmb 0}{\Vert\pmb A\pmb x\Vert_p\over\Vert\pmb x\Vert_p}
\end{align}
$$

In the special cases $$p=1,2,\infty$$, we have

$$
\begin{align}
&\Vert\pmb A\Vert_1=\max_{\pmb x\ne \pmb 0}{\sum_i|\sum_j A_{i,j}x_j|\over\sum_j |x_j|}=\max_{\pmb x\ne\pmb 0}\sum_j {|x_j|\over\sum_k |x_k|}\sum_i|A_{i,j}|=\max_j\sum_i|A_{i,j}|\\\
&\Vert\pmb A\Vert_\infty=\max_{\pmb x\ne \pmb 0}{\max_i|\sum_jA_{i,j}x_j|\over\max_j|x_j|}=\max_i\sum_j|A_{i,j}|\max_{\pmb x\ne \pmb 0}{|x_j|\over\max_k|x_k|}=\max_i\sum_j|A_{i,j}|\\\
&\Vert\pmb A\Vert_2=\max_{\pmb x\ne\pmb 0}\sqrt{\pmb x^\top\pmb A^\top\pmb A\pmb x\over\pmb x^\top\pmb x}=\sigma_1(\pmb A)
\end{align}
$$

where, in the first two derivations, we use $$\max_{\pmb x\ne\pmb 0}\vert \sum_{j}A_{i,j}x_j\vert =\max_{\pmb x\ne\pmb 0}\sum_j\vert A_{i,j}\vert \vert x_j\vert $$ because for $$A_{i,j}x_j<0$$, we can always negate $$x_j$$ to make it positive. Note that $$\Vert\pmb A\Vert_1$$ and $$\Vert\pmb A\Vert_\infty$$ are the maximum absolute column and row sums in $$\pmb A$$, respectively. The last equation is obtained from the Rayleigh quotients, where $$\sigma_1(\pmb A)$$ is the largest singular value. We often call $$\ell_2$$ norm the **spectral norm**.

**Proposition.** (**submultiplicative**) $$\Vert \pmb A\pmb x\Vert_p\le\Vert\pmb A\Vert_p\Vert\pmb x\Vert_p$$, $$\Vert \pmb A\pmb B\Vert_p\le\Vert\pmb A\Vert_p\Vert\pmb B\Vert_p$$

Another matrix norm is **Frobenius norm**, defined as

$$
\begin{align}
\Vert\pmb A\Vert_F=\sqrt{\sum_i\sum_jA_{i,j}^2}=\sqrt{tr(\pmb A^\top\pmb A)}=\sqrt{\sum_i\sigma_i^2(\pmb A)}
\end{align}
$$

A matrix norm $$\Vert\cdot\Vert$$ is said to be **unitary invariant** if

$$
\begin{align}
\Vert\pmb U\pmb A\pmb V\Vert=\Vert\pmb A\Vert
\end{align}
$$

for all *orthogonal* $$\pmb U$$ and $$\pmb V$$ of appropriate size. Unitary invariant norms essentially depend only on the singular values of a matrix

$$
\begin{align}
\Vert\pmb A\Vert=\Vert \pmb U^\top\pmb A\pmb V\Vert=\Vert \pmb U^\top\pmb U\pmb \Sigma\pmb V^\top\pmb V\Vert =\Vert\pmb\Sigma\Vert
\end{align}
$$

where $$\pmb U\pmb \Sigma\pmb V^\top$$ is the singular decomposition of $$\pmb A$$.

**Proposition.** The spectral norm and the Frobenius norm are unitary invariant.

**Proof.** For the Frobenius norm, we have

$$
\begin{align}
\Vert\pmb U\pmb A\pmb V\Vert_F=\sqrt{tr\big((\pmb U\pmb A\pmb V)^\top\pmb U\pmb A\pmb V\big)}=\sqrt{tr(\pmb V^\top\pmb A^\top\pmb A\pmb V)}=\sqrt{tr(\pmb V\pmb V^\top\pmb A^\top\pmb A)}=\sqrt{tr(\pmb A^\top\pmb A)}=\Vert\pmb A\Vert_F
\end{align}
$$

For the spectral norm, 

$$
\begin{align}
\Vert\pmb U\pmb A\pmb V\Vert_2=\max_{\pmb x\ne \pmb 0}{\Vert\pmb U\pmb A\pmb V^\top\pmb x\Vert_2\over\Vert\pmb x\Vert_2}=\max_{\pmb x\ne \pmb 0}{\Vert\pmb A\pmb V^\top\pmb x\Vert_2\over\Vert\pmb x\Vert_2}=\max_{\pmb x\ne \pmb 0}{\Vert\pmb A\pmb y\Vert_2\over\Vert\pmb y\Vert_2}=\Vert \pmb A\Vert_2
\end{align}
$$

where in the second step, we have used $$\Vert \pmb U\pmb x\Vert_2=\Vert\pmb x\Vert_2$$ for any orthogonal $$\pmb U$$, and in the third step, we have used the change of variable $$\pmb y=\pmb V^\top\pmb x$$, which satisfies $$\Vert\pmb y\Vert_2=\Vert\pmb x\Vert_2$$ as $$\pmb V$$ is orthogonal.

## Low-rank approximation

**Theorem** (*Eckar-Young-Mirsky*) Let $$\Vert\cdot\Vert$$ be a *unitary invariant matrix norm*. Suppose $$\pmb A\in\mathbb R^{m \times n}$$, where $$m\ge n$$, has singular value decomposition $$\pmb A=\sum_{i=1}^n\sigma_i\pmb u_i\pmb v_i^\top$$. Then the best rank-k approximation to $$\pmb A$$, where $$k\le \text{rank}(\pmb A)$$, is given by

$$
\begin{align}
\pmb A_k=\sum_{i=1}^k\sigma_i\pmb u_i\pmb v_i
\end{align}
$$

In the sense that 

$$
\begin{align}
\Vert\pmb A-\pmb A_k\Vert\le\Vert \pmb A-\tilde{\pmb A}\Vert
\end{align}
$$

For any $$\tilde{\pmb A}\in\mathbb R^{m \times n}$$ with $$\text{rank}(\tilde{\pmb A})\le k$$.

**Proof.** We show the proof for the spectral norm.

Assuming $$\text{rank}(\tilde{\pmb A})=k$$, we have

$$
\begin{align}
\dim\text{null}(\tilde{\pmb A})+\dim\text{rank}(V_{k+1})=n+1
\end{align}
$$

where $$V_{k+1}=\{\pmb v_1,\dots,\pmb v_{k+1}\}$$ is the first $$k+1$$ right singular vectors of $$\pmb A$$. Therefore, there exists

$$
\begin{align}
\pmb x\in\text{null}(\tilde{\pmb A})+\text{rank}(V_{k+1}),\quad\Vert\pmb x\Vert_2=1
\end{align}
$$

Hence

$$
\begin{align}
\Vert \pmb A-\tilde{\pmb A}\Vert_2^2\ge&\Vert(\pmb A-\tilde{\pmb A})\pmb x\Vert_2^2&\color{red}{\text{by definition of the matrix norm}}\\\
=&\Vert\pmb A\pmb x\Vert_2^2& \color{red}{\text{since }x\in \text{null}(\tilde{\pmb A})}\\\
=&\sum_{i=1}^{k+1}\sigma_i^2|\pmb v_i^\top\pmb x|^2&\color{\red}{\text{since }x\in\text{rank}(V_{k+1})}\\\
\ge&\sigma_{k+1}^2\sum_{i=1}^{k+1}|\pmb v_i^\top\pmb x|^2\\\
=&\sigma_{k+1}^2&\color{red}{\text{since }\pmb v_i\text{ is orthonormal vectors and }\Vert\pmb x\Vert_2=1}
\end{align}
$$

where $$\sigma_{i}$$ is the $$i^{th}$$ singular value. Because, by the definition of the matrix norm, $$\Vert \pmb A-\pmb A_k\Vert_2=\sigma_{k+1}^2$$, $$\Vert\pmb A-\pmb A_k\Vert_2\le\Vert \pmb A-\tilde{\pmb A}\Vert_2$$.

## Pseudoinverses

Let $$\pmb A\in\mathbb R^{m \times n}$$. If $$m\ne n$$, then $$\pmb A$$ is not invertible. **Moore-Penrose pseudoinverse** generalizes the notion of the inverse and defines $$\pmb A^\dagger\in\mathbb R^{n\times m}$$, which always exists and is defined uniquely by the following properties.

1. $$\pmb A\pmb A^\dagger\pmb A=\pmb A$$
2. $$\pmb A^\dagger\pmb A\pmb A^\dagger=\pmb A^\dagger$$
3. $$\pmb A\pmb A^\dagger$$ is symmetric
4. $$\pmb A^\dagger\pmb A$$ is symmetric

We can compute the pseudoinverse of $$\pmb A$$ from its singular value decomposition: if $$\pmb A=\pmb U\pmb \Sigma\pmb V^\top$$, then

$$
\begin{align}
\pmb A^\dagger=\pmb V\pmb\Sigma^\dagger\pmb U^\top
\end{align}
$$

Where $$\pmb\Sigma^\dagger$$ can be computed from $$\pmb\Sigma$$ by taking the transpose(so $$\pmb \Sigma^\dagger\in\mathbb R^{n\times m}$$) and inverting the nonzero singular values on the diagonal.

## References

Thomas, Garrett. 2018. “Mathematics for Machine Learning” 56 (5): 1–47.