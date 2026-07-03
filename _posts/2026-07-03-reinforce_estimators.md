---
layout: post
title: REINFORCE, Bayesian Experimental Design, and What Gradient We Are Really Taking
date: 2026-07-02 15:09:00
description: Exploring the difference between reinforcement learning applied to Bayesian experimental design and REINFORCE-style gradients used inside standard Bayesian experimental design objectives
tags: machine-learning reinforcement-learning bayesian-experimental-design gradients information-gain
categories: Theory
featured: true
---

# REINFORCE, Experimental Design, and What Gradient We Are Really Taking

Bayesian experimental design and reinforcement learning both address sequential decision-making under uncertainty, and the connection between them is natural. In sequential Bayesian experimental design, we choose a design, observe an outcome, update our beliefs, and choose the next design, a loop that looks very much like the kind of sequential decision problem reinforcement learning is built for.

Several papers make this connection explicit. Blau et al. (2022) formulate sequential Bayesian experimental design as a hidden-parameter MDP and use deep reinforcement learning to learn design policies, motivated partly by the fact that RL can handle settings where the design space is discrete, the probabilistic model is a black box, or the likelihood is non-differentiable (implicit). DAD (foster et al. (2021)), and later variants such as iDAD (Ivanova et al. (2021)) and Step-DAD (Hedman et al. (2025)), also train amortized design policies, but instead do so by directly differentiating information-theoretic objectives such as contrastive lower bounds on expected information gain.

So it is tempting to say: “these are all just policy-gradient methods for expected information gain.”

But that hides an important distinction. The RL formulation and the DAD-style score-function estimator target polcies that perform well under the sequential PCE lower bound, but ultimately are taking distinct gradients that answer different conceptual questions.

The difference comes down to whether the reward is treated as an external scalar, or as a differentiable function of the policy parameters.

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

I have written $R_\phi(h)$, rather than just $R(h)$, because in Bayesian experimental design the reward is often not an external scalar. It may be a likelihood ratio, a contrastive information bound, or some other differentiable function of the design.

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

This general form contains two terms, and they differ in what they hold fixed.

The direct term,

$$
\nabla_\phi R_\phi(h),
$$

holds the sampled outcomes fixed but lets the designs vary with $\phi$. It asks:

> If we regenerate the designs in this history using a slightly different policy, but the outcomes are fixed, how does the reward change?

This is a pathwise, gradient: it flows through the explicit $\phi$-dependence inside $R_\phi$.

The score-function term,

$$
R_\phi(h)\nabla_\phi\log p_\phi(h),
$$

holds the entire trajectory fixed, both the designs and the outcomes. It asks:

> Leaving this exact history fixed, designs and outcomes alike, should the policy make a trajectory like this more or less likely to occur in the first place?

This distinction matters in Bayesian experimental design because a design choice affects two separate things: how informative a given observation turns out to be, and how likely that observation was to occur under the sampled parameter. The direct term captures the first effect, adjusting the designs so that the outcome we actually got becomes more informative. The score-function term captures the second, reweighting entire trajectories so that those with high information reward become more probable, and those with low information reward become less probable, without changing how informative any single trajectory is judged to be.

The two terms split along a line of differentiability. The policy's designs are a differentiable function of $\phi$, so the direct term can differentiate through them. The sampled outcomes are not, because we are assuming a likelihood that cannot be reparameterized, so there is no differentiable path from $\phi$ through the outcome. What remains differentiable is the log-probability the model assigns to the outcome that occurred, and this is what the score-function term uses.

For example, if $y_t$ is discrete, there is no differentiable path from $\phi$ through the realized sample $y_t$. But

$$
\log p_\phi(y_t)
$$

is differentiable. The score-function term works with this log-probability rather than with $y_t$ itself, which is how it shifts probability mass toward better trajectories without differentiating through the outcome.


## Why Standard RL Looks Simpler

In most reinforcement learning presentations, the reward is assumed to come from the environment. The policy parameters affect which trajectories are sampled, but the reward assigned to a realized trajectory is treated as fixed once the trajectory has occurred.

That is, the objective is written as

$$
J_{\mathrm{RL}}(\phi)
=
\mathbb E_{h\sim p_\phi(h)}
\left[
R(h)
\right],
$$

where $R$ has no direct dependence on $\phi$.

Then

$$
\nabla_\phi R(h)=0,
$$

and the general identity reduces to

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

This is the familiar REINFORCE estimator.

A baseline can be subtracted because

$$
\mathbb E_{h\sim p_\phi}
\left[
b\nabla_\phi\log p_\phi(h)
\right]
=
b
\int p_\phi(h)\nabla_\phi\log p_\phi(h)\,dh
=
b
\int \nabla_\phi p_\phi(h)\,dh
=
b\nabla_\phi 1
=
0.
$$

This holds only when $b$ does not depend on $\phi$. The baseline may depend on quantities observed before the action, such as the state or the history prefix, but if it depends on $\phi$ then $\nabla_\phi$ acts on it too and the term no longer vanishes. Subject to that, one usually writes

$$
\boxed{
\nabla_\phi J_{\mathrm{RL}}(\phi)
=
\mathbb E_{h\sim p_\phi}
\left[
(R(h)-b)\nabla_\phi\log p_\phi(h)
\right].
}
$$

The above is a special case of the general identity under the assumption that the reward does not directly depend on the policy parameters.

## The Bayesian Experimental Design Setting

In sequential Bayesian experimental design, the policy chooses designs,

$$
\xi_t = \pi_\phi(h_{t-1}),
$$

then the model produces observations,

$$
y_t \sim p(y_t\mid \theta,\xi_t).
$$

The history is

$$
h_T
=
\{(\xi_1,y_1),\dots,(\xi_T,y_T)\}.
$$

DAD optimizes a contrastive lower bound on expected information gain. Given latent samples $\theta_0,\dots,\theta_L$, define

$$
p_\ell(h_T)
=
p(h_T\mid \theta_\ell,\pi_\phi).
$$

The sPCE reward-like quantity is

$$
R_\phi(h_T,\theta_{0:L})
=
\log
\frac{
p_0(h_T)
}{
\sum_{\ell=0}^L p_\ell(h_T)
}.
$$

The objective is

$$
L_T(\phi)
=
\mathbb E_{\theta_{0:L}}
\mathbb E_{h_T\sim p_0}
\left[
R_\phi(h_T,\theta_{0:L})
\right].
$$

This looks like an RL objective, but the reward is not just an external scalar. It is a likelihood-ratio expression that depends on the policy through the designs inside the likelihood terms.

Applying the general identity gives

$$
\nabla_\phi L_T
=
\mathbb E
\left[
R_\phi
\nabla_\phi\log p_0(h_T)
+
\nabla_\phi R_\phi
\right].
$$

Now expand the direct reward gradient:

$$
\nabla_\phi R_\phi
=
\nabla_\phi\log p_0(h_T)
-
\nabla_\phi
\log
\sum_{\ell=0}^L p_\ell(h_T).
$$

The first term has zero expectation under $h_T\sim p_0$:

$$
\mathbb E_{h_T\sim p_0}
\left[
\nabla_\phi\log p_0(h_T)
\right]
=
0.
$$

Therefore, in expectation,

$$
\nabla_\phi L_T
=
\mathbb E
\left[
R_\phi
\nabla_\phi\log p_0(h_T)
-
\nabla_\phi
\log
\sum_{\ell=0}^L p_\ell(h_T)
\right].
$$

Substituting the definition of $R_\phi$,

$$
\boxed{
\nabla_\phi L_T
=
\mathbb E
\left[
\left(
\log
\frac{
p(h_T\mid\theta_0,\pi_\phi)
}{
\sum_{\ell=0}^L p(h_T\mid\theta_\ell,\pi_\phi)
}
\right)
\nabla_\phi
\log p(h_T\mid\theta_0,\pi_\phi)
-
\nabla_\phi
\log
\sum_{\ell=0}^L
p(h_T\mid\theta_\ell,\pi_\phi)
\right].
}
$$

With a baseline $b$, the first term can be replaced by

$$
(R_\phi-b)\nabla_\phi\log p_0(h_T),
$$

giving

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
\right].
}
$$

This is the DAD-style score-function estimator for the sPCE objective. It contains both the RL-like probability-shifting term and a direct likelihood-gradient term.

## RL Applied to BED: The Blau et al. (2022) Perspective

Blau et al. (2022) take a different route. Instead of directly differentiating the sPCE expression through a deterministic design network, they construct a hidden-parameter MDP whose expected return is equal to a sequential experimental design objective based on sPCE. They then solve this MDP with modern deep RL methods.

Abstractly, their RL objective is

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

The reward is constructed so that the expected return corresponds to the same contrastive design objective. But once this is cast as an RL problem, the policy-gradient update is of the standard form

$$
\nabla_\phi
\mathbb E_{\tau\sim p_\phi(\tau)}
\left[
\sum_t r_t
\right],
$$

estimated using terms like

$$
\mathbb E
\left[
Q^\pi(s_t,a_t)
\nabla_\phi\log\pi_\phi(a_t\mid s_t)
\right].
$$

Conceptually, this asks:

> Which sampled actions or designs led to high long-run sPCE reward, and should the policy make those actions more likely?

That is the standard RL question. The estimator upweights sampled design trajectories that produced high contrastive information reward and downweights those that did not.

But it does not ask the DAD pathwise question:

> For this same sampled outcome path, how would a small differentiable change in the design alter the likelihood-ratio reward?

That is the key distinction. In the pure RL setup, the policy is trained to place more probability mass on trajectories that lead to high sPCE return. In DAD, when the model is differentiable through the design, the estimator can also exploit how changing the design infinitesimally changes the likelihood-ratio reward itself.

So the difference is not merely “BED versus RL.” Both can target an sPCE-style experimental design objective. The difference is the gradient being estimated:

$$
\text{RL policy gradient:}
\qquad
\text{make high-sPCE sampled trajectories more likely.}
$$

$$
\text{DAD score/pathwise gradient:}
\qquad
\text{make high-sPCE histories more likely, and change the design to improve the likelihood-ratio reward.}
$$

This is why it is important to be precise about the phrase “EIG gradient.” At finite $L$, both methods are generally optimizing a contrastive lower-bound surrogate, not the exact expected information gain itself. And the RL formulation, although equivalent at the level of expected return, uses a policy-gradient estimator that treats the reward as an RL return. It is therefore primarily reweighting trajectories according to their sPCE return.

DAD’s REINFORCE-style estimator is different: it uses the score-function term for non-reparameterizable outcomes, but still exploits differentiability of the reward with respect to the policy-produced designs.

## Takeaway

The apparent link between reinforcement learning and Bayesian experimental design is real, but there are two distinct uses of “REINFORCE-style” gradients.

In a pure RL formulation, the reward is treated as an external return. The gradient asks which sampled actions led to high reward and should be made more likely.

In DAD-style Bayesian experimental design, the reward is often a differentiable likelihood-ratio objective. The gradient contains the usual score-function term, but also a direct term asking how the likelihood-ratio reward would change if the design itself were perturbed.

That is the conceptual difference:

$$
\boxed{
\text{RL applied to BED: upweight good sampled design trajectories.}
}
$$

$$
\boxed{
\text{DAD-style BED gradient: upweight good histories and differentiate the information objective through the design.}
}
$$

This also explains why the DAD estimator looks more complex than the usual REINFORCE estimator. The usual RL expression is the special case where the reward does not directly depend on the policy parameters. In Bayesian experimental design, that assumption often fails in exactly the interesting way.

## References

Blau, T., Bonilla, E. V., Chades, I., & Dezfouli, A. (2022). *Optimizing Sequential Experimental Design with Deep Reinforcement Learning*. Proceedings of the 39th International Conference on Machine Learning, PMLR 162.  
[https://proceedings.mlr.press/v162/blau22a.html](https://proceedings.mlr.press/v162/blau22a.html)

Foster, A., Ivanova, D. R., Malik, I., & Rainforth, T. (2021). *Deep Adaptive Design: Amortizing Sequential Bayesian Experimental Design*. Proceedings of the 38th International Conference on Machine Learning, PMLR 139.  
[https://proceedings.mlr.press/v139/foster21a.html](https://proceedings.mlr.press/v139/foster21a.html)

Hedman, M., Ivanova, D. R., Guan, C., & Rainforth, T. (2025). *Step-DAD: Semi-Amortized Policy-Based Bayesian Experimental Design*. Proceedings of the 42nd International Conference on Machine Learning, PMLR 267:22904–22923.  
[https://proceedings.mlr.press/v267/hedman25a.html](https://proceedings.mlr.press/v267/hedman25a.html)

Ivanova, D. R., Foster, A., Kleinegesse, S., Gutmann, M. U., & Rainforth, T. (2021). *Implicit Deep Adaptive Design: Policy-Based Experimental Design without Likelihoods*. Advances in Neural Information Processing Systems 34.  
[https://papers.nips.cc/paper/2021/hash/d811406316b669ad3d370d78b51b1d2e-Abstract.html](https://papers.nips.cc/paper/2021/hash/d811406316b669ad3d370d78b51b1d2e-Abstract.html)

