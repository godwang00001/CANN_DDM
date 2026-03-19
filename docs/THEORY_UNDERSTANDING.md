# Theory Understanding

## Status
This file records my current understanding of the theory in the latest manuscript source `docs/main.tex`. It is a theory checkpoint for user review, not code documentation. It should track the manuscript logic, equations, and intended mechanism rather than older implementation assumptions.

## Core Theory
The manuscript proposes a biologically plausible circuit realization of evidence accumulation in which a one-dimensional decision variable $x(t) \in [0,1]$ is encoded as a position shift along a neural manifold rather than as a pure amplitude change. The key claim is that two experimentally observed response types in decision tasks, ramping neurons and sequential neurons, can be understood as two population geometries that encode the same latent decision variable through a shared neural coordinate $\hat{\theta}(x)$.

At the abstract level, the target computation is a pulse-based drift-diffusion process,

$$
dx = v(t)\,dt + c\,dW,
$$

where right and left cues correspond to $v(t)=+v$ and $v(t)=-v$, respectively. The circuit is designed so that cue-dependent motion of the aligned edge-bump state reproduces the corresponding updates in latent decision-variable space.

## Population Geometries
### Ramping Population as an Edge Code
Each ramping neuron is assigned a coordinate $\theta$ in a one-dimensional neural space and has an exponential tuning curve over decision-variable space:

$$
f_E^\theta(x) = \exp[-\lambda_E(\theta)(1-x)].
$$

The ramping-rate function varies geometrically across neural coordinate:

$$
\lambda_E(\theta) = A e^{\gamma \theta}.
$$

In the continuum limit this gives the population response

$$
\rho_E(\theta \mid x) = \exp[-A(1-x)e^{\gamma \theta}],
$$

which has a double-exponential edge-like shape. The informative variable is the edge position, defined as the location of the steepest transition:

$$
\hat{\theta}(x) := \arg\max_\theta \left| \partial_\theta \rho_E(\theta \mid x) \right|.
$$

The manuscript derives the closed-form map

$$
\hat{\theta}(x) = -\frac{1}{\gamma}\left[\ln A + \ln(1-x)\right].
$$

So the decision variable is encoded logarithmically as a translation of the edge along neural space. The same population response can then be rewritten as a translated profile,

$$
\rho_E(\theta \mid x) = \exp[-\lambda_E(\theta-\hat{\theta}(x))].
$$

For later analysis, the manuscript replaces this double-exponential edge by a sigmoid approximation:

$$
\rho_E(\theta \mid x) \approx \sigma\bigl(\gamma'(\theta-\hat{\theta}(x))\bigr),
$$

with $\sigma(z)=1/(1+e^{-z})$ and $\gamma' = 4\gamma/e$, chosen to match the local slope of the target edge profile. My understanding is that this sigmoid is the main analytical object used in the later circuit and perturbation arguments.

### Sequential Population as a Bump Code
The sequential population is modeled as a Gaussian bump whose center follows the same $\hat{\theta}(x)$:

$$
\rho_B(\theta \mid x) = \exp\left[-\frac{(\theta-\hat{\theta}(x))^2}{2a^2}\right].
$$

So the sequential code is not a separate latent variable. It is another geometric population realization of the same decision state, aligned to the same neural position as the edge.

The manuscript also makes the single-neuron sequential tuning explicit. If neuron $\theta$ has preferred decision value $x_B$ defined implicitly by $\hat{\theta}(x_B)=\theta$, then its tuning over decision-variable space is

$$
f_B^\theta(x)
=
\exp\left[
-\frac{1}{2a^2\gamma'^2}
\left(
\ln \frac{1-x}{1-x_B}
\right)^2
\right].
$$

This means the sequential neuron tuning is a log-Gaussian over $x$, not a simple Gaussian on a linear decision axis. My current understanding is that this is important conceptually: the bump code inherits the same nonlinear warping as the ramping code.

## Coupled Edge-Bump Circuit
The circuit contains two interacting rate-based continuous attractor populations:

- an edge population for ramping-like activity
- a bump population for sequential-like activity

Their dynamics are written as coupled neural fields with recurrent kernels $W_{EE}$ and $W_{BB}$ and pointwise nonlinearities $\phi_E$ and $\phi_B$. The recurrent connectivity is translation-invariant in the interior and is designed so that the edge population supports translated edge attractors and the bump population supports translated bump attractors.

At the theory level:

- the edge attractor is supported by a difference-of-Gaussians style recurrent kernel and a sigmoidal nonlinearity
- the bump attractor is supported by a Gaussian kernel and a quadratic nonlinearity

The purpose of these recurrent populations is to preserve shape while allowing position to move along the manifold.

## Reciprocal Coupling Logic
### Edge-to-Bump Pathway
The edge-to-bump pathway is modeled through an interaction kernel $W_{EB}$. In the local neural-field limit, the manuscript argues that this operator can be approximated by a differential operator acting on the edge activity, and the leading term is chosen to be

$$
I_{EB}(\theta) \propto \partial_\theta r_E(\theta).
$$

Because the derivative of an edge is a localized bump-shaped profile centered at the edge location, the edge population generates a bump-shaped drive whose center indicates the current encoded state. If the internal bump is offset from that location, this input produces bump-tracking dynamics that pull the bump toward the edge-implied center.

### Bump-to-Edge Pathway
For the bump-to-edge interaction, the manuscript takes a purely local coupling limit,

$$
W_{BE}(\theta,\theta') = c_{BE}\,\delta(\theta-\theta'),
$$

so that

$$
I_{BE}(\theta) = c_{BE} r_B(\theta).
$$

The manuscript’s intended mechanism is that a centered positive or negative bump input translates the edge rightward or leftward without changing its shape. So the bump does not merely mirror the edge. It supplies the shape-preserving drive that moves the encoded edge state.

## Perfect-Alignment Regime
The key dynamical regime is what the manuscript calls perfect alignment. In this regime, bump tracking driven by $I_{EB}$ is sufficiently fast relative to the intrinsic edge motion driven by $I_{BE}$ that the bump stays effectively aligned with the edge-implied location.

My current understanding is that this means the coupled dynamics can be treated as motion of a single aligned edge-bump state. Operationally, the circuit evolution is:

$$
I_{BE} \rightarrow r_E \rightarrow I_{EB} \rightarrow r_B.
$$

A cue first changes the bump-to-edge input, that shifts the edge in neural space, the displaced edge generates a new derivative-like input to the bump population, and the bump then tracks the new edge position while remaining aligned. Right cues and left cues therefore induce opposite translations of the same aligned attractor configuration.

## Evidence Update Mechanism
Because the map from decision-variable space to neural-position space is nonlinear, a constant increment in $x$ does not correspond to a constant shift in $\theta$. The manuscript therefore derives a position-dependent bump-to-edge coupling that corrects for this nonlinear geometry.

Given the logarithmic map, a cue that produces a constant increment in latent decision space implies a position-dependent neural displacement:

$$
\Delta\theta(\theta)
=
\begin{cases}
\dfrac{1}{\gamma'} \ln\left(\dfrac{1}{1-Ave^{\gamma'\theta}}\right), & \text{right cue} \\
-\dfrac{1}{\gamma'} \ln\left(\dfrac{1}{1-Ave^{\gamma'\theta}}\right), & \text{left cue.}
\end{cases}
$$

In the linear-response regime, the cue-evoked edge displacement is approximately proportional to bump amplitude:

$$
|\Delta\theta| \approx \kappa c_{BE},
$$

where $\kappa$ is the effective gain from bump amplitude to edge displacement.

Combining these gives the manuscript’s position-dependent coupling law:

$$
c_{BE}(\theta)
\approx
\frac{1}{\kappa \gamma'}
\ln\left(\frac{1}{1-Ave^{\gamma'\theta}}\right).
$$

For small cue size this simplifies to

$$
c_{BE}(\theta) \propto e^{\gamma' \theta}.
$$

The evidence-modulated bump-to-edge drive is then specified as

$$
I_{BE}(\theta,t) = v(t)c_{BE}(\theta) r_B(\theta).
$$

Under the perfect-alignment assumption, the manuscript claims this lets each cue move the aligned edge-bump state by exactly the neural displacement corresponding to the target increment in latent decision-variable space.

## Appendix-Level Speed Interpretation
The appendix makes the velocity statement more explicit. In the weak-input regime, the edge activity is assumed to stay near the translated edge manifold,

$$
r_E(\theta,t) \approx r_E^*(\theta-\hat{\theta}(t)),
$$

and projection onto the positional-shift mode yields a leading-order speed law of the form

$$
\dot{\hat{\theta}}(t)
=
\frac{c_{BE}}{\tau_E}
\frac{
\int \partial_\theta r_E^*(\theta)\rho_B(\theta)\,d\theta
}{
\int \frac{(\partial_\theta r_E^*(\theta))^2}{D(\theta)}\,d\theta
}.
$$

My understanding is that this appendix result is the theoretical justification for treating edge velocity as approximately linear in $c_{BE}$ in the small-signal regime, which is exactly the relation we want to calibrate numerically.

## Current Theory Summary
My current understanding of the latest manuscript is:

- the latent decision variable is represented as a translated position $\hat{\theta}(x)$ on a neural manifold
- ramping neurons form an edge code derived from Laplace-like exponential single-neuron tunings
- sequential neurons form an aligned bump code with log-Gaussian single-neuron tuning over decision-variable space
- the recurrent edge and bump populations are designed as coupled continuous attractors that preserve profile shape while allowing translation
- the derivative-like edge-to-bump pathway maintains bump alignment with the edge
- the local bump-to-edge pathway is the mechanism that actually moves the encoded state
- because the $x \mapsto \hat{\theta}$ map is nonlinear, the required $c_{BE}$ must depend on position
- in the small-signal regime this yields an approximately exponential $c_{BE}(\theta)$ profile with exponent $\gamma'$

## Open Questions
The current manuscript-level questions I still see are:

- how the manuscript’s abstract stochastic term $c\,dW$ should eventually be instantiated in the full circuit dynamics rather than only at the target-process level
- how broadly the perfect-alignment assumption holds outside the weak-input / linear-response regime
- how closely the finite-size, bounded simulation used in code will match the idealized continuum arguments used in the manuscript
