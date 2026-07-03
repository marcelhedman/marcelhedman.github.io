---
layout: post
title: REINFORCE, Bayesian Experimental Design, and What Gradient We Are Really Taking
date: 2026-07-02 15:09:00
description: Exploring reinforcement learning applied to Bayesian experimental design and REINFORCE-style gradients pre-existing within Bayesian experimental design objectives
tags: machine-learning reinforcement-learning bayesian-experimental-design gradients information-gain
categories: Theory
featured: true
---

Bayesian experimental design and reinforcement learning both address sequential decision-making under uncertainty, and the connection between them is natural. In sequential Bayesian experimental design, we choose a design, observe an outcome, update our beliefs, and choose the next design, a loop that looks very much like the kind of sequential decision problem reinforcement learning is built for.

Several papers make this connection explicit. Blau et al. (2022) formulate sequential Bayesian experimental design as a hidden-parameter MDP and use deep reinforcement learning to learn design policies, motivated partly by the fact that RL can handle settings where the design space is discrete, the probabilistic model is a black box, or the likelihood is non-differentiable and implicit. DAD (Foster et al. (2021)), and later variants such as iDAD (Ivanova et al. (2021)) and Step-DAD (Hedman et al. (2025)), also train amortized design policies, and reach for REINFORCE-style score-function estimators to optimize information-theoretic objectives such as contrastive lower bounds on expected information gain.

Despite the clear similarities between the fields, when reading the DAD paper, its gradient estimator for the sPCE objective may at first glance look like a distinct, more complicated object than the plain REINFORCE estimator. Specifically, carrying an extra term that has no counterpart in the familiar REINFORCE estimators from RL. It is easy to come away thinking DAD is optimizing something structurally different.
 
It is not. That estimator is simply the *deterministic-policy* form of the exact same REINFORCE gradient. Swap the deterministic design network for a stochastic, non-reparameterized policy and the extra term vanishes, leaving precisely the standard REINFORCE estimator we are all familiar with. The RL formulation and the DAD-style estimator are not taking different gradients of the objective; they are taking the *same* gradient of the *same* contrastive sPCE objective. The only thing that differs is which terms of that gradient survive, and that is governed entirely by one choice: whether the design policy is deterministic or stochastic.
 
Everything else in this post is unpacking that one fact.


---

## The General Score-Function Identity

Start with the general objective

$$
J(\phi)
=
\mathbb E_{h \sim p_\phi(h)}
\left[
R_\phi(h)
\right],
$$

where $h$ is a sampled history or trajectory, $p_\phi(h)$ is the distribution over histories induced by the current design policy, and $R_\phi(h)$ is the reward-like quantity assigned to that history.

I have written $R_\phi(h)$, rather than just $R(h)$, to allow for the possibility that the reward itself depends on the policy parameters. This happens either because the reward takes $\phi$ as a direct input (rare), or because $\phi$ enters through the designs the reward is evaluated on (via deterministic policies). Keeping the subscript lets us derive the gradient once, in full generality, and then see later exactly when that dependence is present and when it drops away.

Writing the expectation as an integral,

$$
J(\phi)
=
\int p_\phi(h)R_\phi(h)\,dh.
$$

Differentiating,

$$
\nabla_\phi J(\phi)
=
\int
\nabla_\phi
\left[
p_\phi(h)R_\phi(h)
\right]dh.
$$

By the product rule,

$$
\nabla_\phi J(\phi)
=
\int
R_\phi(h)\nabla_\phi p_\phi(h)\,dh
+
\int
p_\phi(h)\nabla_\phi R_\phi(h)\,dh.
$$

Now use the log-derivative identity,

$$
\nabla_\phi p_\phi(h)
=
p_\phi(h)\nabla_\phi\log p_\phi(h).
$$

Substituting gives

$$
\nabla_\phi J(\phi)
=
\int
p_\phi(h)
R_\phi(h)\nabla_\phi\log p_\phi(h)\,dh
+
\int
p_\phi(h)\nabla_\phi R_\phi(h)\,dh.
$$

Therefore,

$$
\boxed{
\nabla_\phi J(\phi)
=
\mathbb E_{h\sim p_\phi}
\left[
R_\phi(h)\nabla_\phi\log p_\phi(h)
+
\nabla_\phi R_\phi(h)
\right].
}
$$


This general form contains two terms, but they do not both appear in every case. Which of them is present is decided by how the designs depend on $\phi$, and that is a property of the design policy.

The score-function term,
 
$$
R_\phi(h)\nabla_\phi\log p_\phi(h),
$$
 
is always present. It comes from the fact that $\phi$ shapes the distribution over trajectories $p_\phi(h)$. It asks:
 
> Leaving this exact history fixed, designs and outcomes alike, should the policy make a trajectory like this more or less likely to occur in the first place?

This is the only route through which a *stochastic, non-reparameterized* policy can be trained. When the designs are sampled from the policy, they are not differentiable functions of $\phi$; the only place $\phi$ appears in a realized trajectory is in the probability the policy assigned to the actions it took, i.e. in $\log p_\phi(h)$. So this term reweights whole trajectories, pushing probability toward those with high reward and away from those with low reward, without ever differentiating the reward itself.

The direct term,
 
$$
\nabla_\phi R_\phi(h),
$$
 
is present only when $\phi$ enters the reward directly as an input or pathwise via the design policy. Specifically, this occurs when the designs are a differentiable function of $\phi$, which is exactly the *deterministic* (or reparameterized) policy case. It asks:

> Holding the outcomes fixed, if we nudge the designs the policy would produce, how does the reward change?

This is a pathwise gradient. It does not reweight trajectories; it improves the reward of the trajectory in front of us by adjusting the designs.
 

Formally, the direct term is nonzero only when $\phi$ has a differentiable path into the design $\xi$. For a deterministic policy $\xi_t = \pi_\phi(h_{t-1})$, the design is an explicit differentiable function of $\phi$, so $\nabla_\phi R_\phi \neq 0$. For a stochastic policy whose action is a non-reparameterized sample, that path is severed and $\nabla_\phi R_\phi = 0$. The sampled outcomes, by contrast, are non-differentiable in *either* case, since we are assuming a likelihood that cannot be reparameterized; what remains differentiable is the log-probability the model assigns to the outcome that occurred, and that is what the ever-present score-function term uses.
 

 ---
 
## Why Standard RL Looks Simpler
 
Most reinforcement learning settings assume a stochastic policy with non-reparameterized actions. That single modelling choice is why the estimator seen most often in the literature is the plain REINFORCE score-function estimator with no direct term.
 
We can read this straight off the general identity. With non-reparameterized samples there is no differentiable path from $\phi$ into the actions, so $\nabla_\phi R_\phi = 0$ and only the score-function term survives:

$$
\boxed{
\nabla_\phi J_{\mathrm{RL}}(\phi)
=
\mathbb E_{h\sim p_\phi}
\left[
R(h)\nabla_\phi\log p_\phi(h)
\right].
}
$$

This is the familiar REINFORCE estimator. A baseline $b$ can be subtracted without introducing bias, since

$$
\mathbb E_{h\sim p_\phi}
\left[
b\nabla_\phi\log p_\phi(h)
\right]
=
b
\int \nabla_\phi p_\phi(h)\,dh
=
b\nabla_\phi 1
=
0,
$$

provided $b$ does not depend on $\phi$ (it may depend on the state or history prefix), giving the usual

$$
\nabla_\phi J_{\mathrm{RL}}(\phi)
=
\mathbb E_{h\sim p_\phi}
\left[
(R(h)-b)\nabla_\phi\log p_\phi(h)
\right].
$$



---
 
## DAD: A Deterministic Policy, So Both Terms Survive
 
In sequential Bayesian experimental design the policy chooses designs, $\xi_t = \pi_\phi(h_{t-1})$, and the model produces observations $y_t \sim p(y_t\mid \theta,\xi_t)$, building up a history $h_T = \{(\xi_1,y_1),\dots,(\xi_T,y_T)\}$.
 
DAD optimizes a contrastive lower bound on expected information gain. Given latent samples $\theta_0,\dots,\theta_L$ and writing $p_\ell(h_T) = p(h_T\mid \theta_\ell,\pi_\phi)$, the sPCE reward-like quantity is


$$
R_\phi(h_T,\theta_{0:L})
=
\log
\frac{
p_0(h_T)
}{
\sum_{\ell=0}^L p_\ell(h_T)
},
$$
 
and the objective is $L_T(\phi) = \mathbb E_{\theta_{0:L}}\,\mathbb E_{h_T\sim p_0}\left[R_\phi\right]$.

The decisive fact is that DAD uses a *deterministic* design network: $\xi_t = \pi_\phi(h_{t-1})$ is a differentiable map, so $\phi$ enters $R_\phi$ through the designs inside the likelihood terms. This is exactly the case from the general section where both terms are live. Applying the general identity, and using that $\mathbb E_{h_T\sim p_0}[\nabla_\phi\log p_0(h_T)] = 0$ to drop the numerator's score contribution from the direct term, gives

$$
\boxed{
\nabla_\phi L_T
=
\mathbb E
\left[
(R_\phi-b)
\nabla_\phi\log p_0(h_T)
-
\nabla_\phi
\log
\sum_{\ell=0}^L p_\ell(h_T)
\right],
}
$$

where $b$ is an optional $\phi$-independent baseline. This is precisely the DAD estimator, and it is nothing more than the general two-term identity specialized to a deterministic policy: the first term is the ever-present score-function term, and the second is the direct term $\nabla_\phi R_\phi$, alive only because the deterministic design map lets $\phi$ flow into the likelihood-ratio reward. Swap in a stochastic, non-reparameterized policy and that second term vanishes, collapsing this back to the plain REINFORCE form of the previous section.


The above recovered the form from the DAD paper via the general REINFORCE form. However, it can equivalently be seen directly from recalling the RL algorithm DDPG's (Deep Deterministic Policy Gradient) actor update:
 


$$
\nabla_\phi J
=
\mathbb E_{s}
\left[
\nabla_\xi Q(\xi)\big|_{\xi=\pi_\phi(s)}\,
\nabla_\phi \pi_\phi(s)
\right],
$$
 
and note that here $Q(\xi)$ is itself an expectation of the sPCE log-ratio reward,
 
$$
Q(\xi)
=
\mathbb E_{\theta_{0:L},\,y\sim p(y\mid\theta_0,\xi)}
\left[
\log
\frac{
p(y\mid\theta_0,\xi)
}{
\sum_{\ell=0}^L p(y\mid\theta_\ell,\xi)
}
\right].
$$
 
So when we expand $\nabla_\xi Q(\xi)$, the design enters both inside the log-ratio and inside the sampling density $p(y\mid\theta_0,\xi)$ the expectation is taken over, and the product rule gives two pieces,
 
$$
\nabla_\xi Q(\xi)
=
\mathbb E_{\theta_{0:L},\,y}
\left[
\log
\frac{
p(y\mid\theta_0,\xi)
}{
\sum_{\ell=0}^L p(y\mid\theta_\ell,\xi)
}
\,\nabla_\xi\log p(y\mid\theta_0,\xi)
+
\nabla_\xi
\log
\frac{
p(y\mid\theta_0,\xi)
}{
\sum_{\ell=0}^L p(y\mid\theta_\ell,\xi)
}
\right].
$$
 
Plugging this into the deterministic-policy-gradient form, with $\xi_\phi = \pi_\phi(s)$,
 
$$
\boxed{
\nabla_\phi J
=
\mathbb E_{\theta_{0:L},\,y}
\left[
\log
\frac{
p(y\mid\theta_0,\xi_\phi)
}{
\sum_{\ell=0}^L p(y\mid\theta_\ell,\xi_\phi)
}
\,\nabla_\xi\log p(y\mid\theta_0,\xi_\phi)\,\nabla_\phi\xi_\phi
+
\nabla_\xi
\log
\frac{
p(y\mid\theta_0,\xi_\phi)
}{
\sum_{\ell=0}^L p(y\mid\theta_\ell,\xi_\phi)
}
\,\nabla_\phi\xi_\phi
\right].
}
$$
 
This is the two-term DAD form. The two pieces are the same score-function and direct terms as before, now visible as a single consequence of differentiating a design-dependent expectation. A stochastic, non-reparameterized policy would sever the $\nabla_\phi\xi_\phi$ path and leave only the score-function estimator of the previous section.












---
 
## RL Applied to BED: The Blau et al. (2022) Perspective
 
Blau et al. (2022) target the same sPCE objective, but unlike DAD, work with stochastic policies. They construct a hidden-parameter MDP whose expected return equals the sPCE objective, then solve it with standard deep RL:
 
$$
J_{\mathrm{RL}}(\pi)
=
\mathbb E_{\pi,\mathrm{env}}
\left[
\sum_t r_t
\right]
=
\mathrm{sPCE}(\pi,L,T).
$$
 
Here, the policy is stochastic and its designs are non-reparameterized samples. That is precisely the case from the general section where the direct term is zero, so the update is the more familiar policy gradient with only the score-function term,
 
$$
\nabla_\phi J_{\mathrm{RL}}
=
\mathbb E_{h\sim p_\phi}
\left[
R(h)\,\nabla_\phi\log p_\phi(h)
\right],
$$
 
which reweights sampled design trajectories by their sPCE return and nothing more. It cannot ask how an infinitesimal change in a design would alter the likelihood-ratio reward, because there is no differentiable path from $\phi$ into a sampled design to carry that question.
 
This is the mirror image of the DAD section. There, the extra term was the deterministic policy gradient of the sPCE reward, the DDPG-shaped object we recovered above. Blau et al. simply operate in the regime where that term does not exist. Same gradient, same objective; the stochastic, non-reparameterized policy switches the direct term off, and the deterministic one switches it back on.
 
 
---
 
## Takeaway
 
There are not two distinct "REINFORCE-style" gradients here, one for RL and one for DAD. There is a single general gradient of the contrastive objective, and everything reduces to which of its terms survive.
 
That gradient always contains the score-function term, which reweights whole trajectories toward higher sPCE reward. It contains a second, pathwise term only when $\phi$ has a differentiable path into the design. A stochastic, non-reparameterized policy severs that path, so the pathwise term is zero and we are left with exactly the standard REINFORCE estimator. A deterministic policy, like DAD's design network, keeps the path intact, so the pathwise term stays and additionally asks how the likelihood-ratio reward would change if the design itself were perturbed.
 
So the whole distinction collapses onto the policy class:
 
$$
\boxed{
\text{Stochastic, non-reparameterized policy: upweight good sampled design trajectories only.}
}
$$
 
$$
\boxed{
\text{Deterministic (or reparameterized) policy: also differentiate the information objective through the design.}
}
$$
 
This also explains why the DAD estimator looks more complex than the usual REINFORCE estimator. It is not that BED objectives are special *per se*; it is that DAD uses a deterministic design network, which keeps the direct term alive. The pure-REINFORCE expression is the special case where the policy is stochastic and non-reparameterized, so the direct term vanishes. The reward's dependence on $\phi$ only matters when there is a differentiable path from $\phi$ into the design to carry it, and that path is exactly what a deterministic or reparameterized policy provides.
 

---

## References

Blau, T., Bonilla, E. V., Chades, I., & Dezfouli, A. (2022). *Optimizing Sequential Experimental Design with Deep Reinforcement Learning*. Proceedings of the 39th International Conference on Machine Learning, PMLR 162.  
[https://proceedings.mlr.press/v162/blau22a.html](https://proceedings.mlr.press/v162/blau22a.html)

Foster, A., Ivanova, D. R., Malik, I., & Rainforth, T. (2021). *Deep Adaptive Design: Amortizing Sequential Bayesian Experimental Design*. Proceedings of the 38th International Conference on Machine Learning, PMLR 139.  
[https://proceedings.mlr.press/v139/foster21a.html](https://proceedings.mlr.press/v139/foster21a.html)

Hedman, M., Ivanova, D. R., Guan, C., & Rainforth, T. (2025). *Step-DAD: Semi-Amortized Policy-Based Bayesian Experimental Design*. Proceedings of the 42nd International Conference on Machine Learning, PMLR 267:22904–22923.  
[https://proceedings.mlr.press/v267/hedman25a.html](https://proceedings.mlr.press/v267/hedman25a.html)

Ivanova, D. R., Foster, A., Kleinegesse, S., Gutmann, M. U., & Rainforth, T. (2021). *Implicit Deep Adaptive Design: Policy-Based Experimental Design without Likelihoods*. Advances in Neural Information Processing Systems 34.  
[https://papers.nips.cc/paper/2021/hash/d811406316b669ad3d370d78b51b1d2e-Abstract.html](https://papers.nips.cc/paper/2021/hash/d811406316b669ad3d370d78b51b1d2e-Abstract.html)

