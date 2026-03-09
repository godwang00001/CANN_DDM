# Theory Understanding

## Status
This file records my current understanding of the user's theory and simulation framework. It is meant to capture the conceptual model, not the code structure. It should be revised whenever the user corrects or deepens the explanation.

## Core Idea
The project aims to build a biologically plausible circuit model for evidence accumulation in decision-making tasks. The abstract computation is a one-dimensional decision variable $x(t) \in [0,1]$ governed by drift-diffusion-like dynamics. The main theoretical move is to represent this latent variable not directly in firing-rate amplitude, but as a position shift along a neural manifold.

Two experimentally observed population types are treated as two geometric realizations of the same latent decision variable:

- a ramping population, represented at the population level as an edge-like activity profile
- a sequential population, represented at the population level as a localized bump

Both population codes are aligned through the same position variable $\hat{\theta}(x)$.

## Neural Geometry
The edge population is motivated by Laplace-coded single-neuron tuning curves. Individual ramping neurons have exponential tuning functions whose rate constants vary systematically across the neural coordinate $\theta$. In the continuum limit, this produces a double-exponential population profile with an edge geometry. The informative quantity is the edge location, not the overall amplitude.

The bump population is modeled as a Gaussian-like localized pattern centered at the same $\hat{\theta}(x)$. Thus, ramping and sequential populations are interpreted as two aligned manifolds encoding the same decision variable.

## Coordinate Domains
The latent decision variable is explicitly restricted to $x \in [0,1]$.

The neural coordinate $\theta$ is not intrinsically restricted to $[-\pi/2,\pi/2]$. In simulation, neurons are placed on a larger symmetric interval $[-A,A]$ with $A > \pi/2$, using a uniform discretization across that interval. However, the representation of the decision variable in neural space is intentionally restricted to the central range $(-\pi/2,\pi/2)$. This constraint, together with the edge mapping in Eq. 4 of the manuscript, determines how the edge parameters relate latent-variable space to neural space.

## Circuit-Level Interpretation
The circuit contains two coupled continuous attractor neural networks:

- an edge attractor for ramping-like activity
- a bump attractor for sequential activity

The recurrent weights are chosen so that each population supports a continuous family of translated attractor states.

For the edge population:

- recurrent connectivity is constructed from a difference-of-Gaussians kernel
- the nonlinearity is sigmoidal
- the target fixed point is an edge profile

For the bump population:

- recurrent connectivity is constructed from a Gaussian kernel
- the nonlinearity is quadratic
- the target fixed point is a Gaussian bump

The purpose of these recurrent circuits is to preserve shape while allowing translation along the neural coordinate.

## Functional Roles of the Two Populations
The bump population is not merely a parallel code. It plays a mechanistically necessary role in moving and maintaining the edge representation.

### Edge-to-Bump Pathway
The edge-to-bump input $I_{EB}$ is designed so that, to leading order, it is proportional to the spatial derivative of the edge profile:

$$
I_{EB}(\theta) \propto \partial_\theta r_E(\theta)
$$

Because the derivative of an edge is a localized bump-shaped signal centered at the edge location, the edge population supplies the bump population with a localized drive indicating where the bump should be. If the bump is misaligned, bump dynamics track this input and move toward the edge-implied location.

### Bump-to-Edge Pathway
The bump-to-edge input is taken in a local-coupling limit:

$$
I_{BE}(\theta) = c_{BE} r_B(\theta)
$$

A centered positive or negative bump input can translate the edge leftward or rightward without deforming its geometry. This is the key mechanistic reason the bump population matters: it provides the shape-preserving drive that moves the edge along its attractor manifold.

## Aligned Bump-Edge State
When bump tracking driven by $I_{EB}$ is fast enough relative to the intrinsic edge dynamics, the bump remains aligned to the edge-implied location. In that regime, the bump and edge form a coupled aligned attractor configuration. This aligned state is the reference configuration used to implement decision-variable updates.

## How Evidence Updates the Circuit State
Evidence is modeled as discrete right and left cue pulses in a pulse-based drift-diffusion process. A fixed increment in decision-variable space does not correspond to a fixed displacement in neural-position space, because the mapping from $x$ to $\hat{\theta}(x)$ is nonlinear.

Therefore, the bump-to-edge coupling strength must depend on position. The manuscript derives a position-dependent $c_{BE}(\theta)$ so that each evidence pulse produces the correct increment in latent decision-variable space even though the corresponding displacement in neural space depends on the current state.

For small cue size, this yields an approximate exponential scaling:

$$
c_{BE}(\theta) \propto e^{s\theta}
$$

The resulting evidence-modulated drive is:

$$
I_{BE}(\theta,t) = [\delta_{\mathrm{right}}(t) - \delta_{\mathrm{left}}(t)] c_{BE}(\theta) r_B(\theta)
$$

Under the alignment assumption, this lets the coupled bump-edge attractor track the intended pulse-based drift-diffusion dynamics.

## Current High-Level Summary
My current understanding is that the theory claims all of the following at once:

- ramping and sequential neurons can be interpreted as two aligned geometric population codes for the same latent decision variable
- Laplace-coded ramping tunings naturally generate an edge representation whose position encodes the decision variable logarithmically
- a coupled bump-edge continuous attractor circuit provides a biologically plausible mechanism for implementing this code dynamically
- evidence-dependent modulation of bump-to-edge coupling allows the circuit to approximate drift-diffusion updates in latent-variable space

## Open Question
The abstract target process includes a noise term, but the current circuit description does not yet specify how stochasticity enters the circuit dynamics. The user indicated that this will be added to both the paper and the simulation later. I should keep track of this gap, because how noise is introduced may matter for both theoretical interpretation and implementation details.
