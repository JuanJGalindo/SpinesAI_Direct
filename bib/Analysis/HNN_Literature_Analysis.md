# Hamiltonian Neural Network Variants: Literature Analysis

---

## PHASE 1: Individual Paper Deep-Dives

---

### 1. Hamiltonian Neural Networks (HNN) --- Greydanus, Dzamba & Yosinski, 2019

**1.1 Core Paradigm & Objective**

Standard neural networks learn to map states to their time derivatives directly, treating the dynamics as an unconstrained regression problem. This means they have no mechanism to enforce conservation laws; small per-step errors accumulate and cause trajectories to drift to physically impossible energy states (spiraling inward/outward in phase space). HNNs solve this by parameterizing the *Hamiltonian function* $H_\theta(q,p)$ itself with a neural network, rather than parameterizing the vector field $\dot{x} = f_\theta(x)$. The dynamics are then *derived* from the learned scalar via Hamilton's equations, which guarantees that the learned quantity $H_\theta$ is exactly conserved along predicted trajectories.

**1.2 Mathematical Formulation**

*Mathematical Model:* The learned vector field is the symplectic gradient of the neural-network Hamiltonian:

$$\dot{q} = \frac{\partial H_\theta}{\partial p}, \quad \dot{p} = -\frac{\partial H_\theta}{\partial q}$$

where $H_\theta: \mathbb{R}^{2n} \to \mathbb{R}$ is a fully-connected network (3 layers, 200 hidden units, tanh activations) outputting a scalar.

*Optimization Objective:* Learn $\theta$ such that the symplectic gradient of $H_\theta$ matches the observed time derivatives of the system.

*Loss Function:*

$$\mathcal{L}_{\text{HNN}} = \left\| \frac{\partial H_\theta}{\partial p} - \dot{q} \right\|_2^2 + \left\| \frac{\partial H_\theta}{\partial q} + \dot{p} \right\|_2^2$$

The partial derivatives $\partial H_\theta / \partial q$ and $\partial H_\theta / \partial p$ are computed via automatic differentiation (in-graph gradient).

**1.3 Data & Training Requirements**

Requires phase-space state pairs $(q, p)$ and their *explicit time derivatives* $(\dot{q}, \dot{p})$. For synthetic tasks, analytic derivatives are used; for real data, finite-difference approximations of the time derivative are computed. Data is continuous phase-space trajectories with known $\dot{x}$ labels.

**1.4 Numerical Solver Interaction**

The integrator is used purely as an *external surrogate* at inference time. Training optimizes point-wise derivative matching---no integrator is embedded in the training loop. At test time, a 4th-order Runge-Kutta integrator (scipy `solve_ivp`, tolerance $10^{-9}$) is used to roll out trajectories from the learned vector field.

---

### 2. Port-Hamiltonian Neural Networks (pHNN) --- Desai, Roberts, Mattheakis, Sondak & Protopapas, 2021

**2.1 Core Paradigm & Objective**

The original HNN assumes a conserved Hamiltonian and therefore *cannot* model systems with energy dissipation (friction/damping) or time-dependent external forcing. Real-world dynamical systems are frequently non-autonomous, featuring explicit time-dependent driving forces and energy loss. pHNNs embed the *port-Hamiltonian formalism* into neural networks, decomposing the dynamics into three independently learnable components: a stationary Hamiltonian $H_{\theta_1}$, a time-dependent forcing field $F_{\theta_2}(t)$, and a dissipation coefficient $\hat{N}_{\theta_3}$.

**2.2 Mathematical Formulation**

*Mathematical Model:* The port-Hamiltonian structure modifies Hamilton's equations as:

$$\begin{pmatrix} \dot{q} \\ \dot{p} \end{pmatrix} = \left[ \begin{pmatrix} 0 & I \\ -I & 0 \end{pmatrix} + \begin{pmatrix} 0 & 0 \\ 0 & N \end{pmatrix} \right] \begin{pmatrix} \partial H / \partial q \\ \partial H / \partial p \end{pmatrix} + \begin{pmatrix} 0 \\ F(t) \end{pmatrix}$$

Three separate neural network components: $H_{\theta_1}(q,p)$ (3 hidden layers, stationary Hamiltonian), $F_{\theta_2}(t)$ (3 hidden layers, force as function of time only), $\hat{N}_{\theta_3}$ (single learnable weight for the damping coefficient).

*Optimization Objective:* Match predicted state-time derivatives to ground truth while regularizing force and damping to encourage parsimony (Occam's razor via L1).

*Loss Function:*

$$\mathcal{L} = \|\hat{\dot{q}}_t - \dot{q}_t\|_2^2 + \|\hat{\dot{p}}_t - \dot{p}_t\|_2^2 + \lambda_F \|\hat{F}_{\theta_2}\|_1 + \lambda_N \|\hat{N}_{\theta_3}\|_1$$

The L1 penalties on force and damping prevent the network from learning spurious non-zero force/damping terms in conservative systems.

**2.3 Data & Training Requirements**

Requires explicit time-derivative labels $(\dot{q}, \dot{p})$ at each data point, generated via an RK4 integrator with $\text{rtol} = 10^{-10}$. The input triple is $[q, p, t]$. The authors also demonstrate an embedded-integrator variant where the loss becomes $\|[\hat{q}, \hat{p}]_{t+1} - [q,p]_{t+1}\|^2$ (state-to-state), showing the method works with both derivative data and discrete trajectory data.

**2.4 Numerical Solver Interaction**

Primary training uses derivative matching (no integrator in the training loop). An alternative variant embeds an RK4 integrator within the training loop for state-to-state prediction. At inference, a time integrator rolls out predictions from test initial conditions.

---

### 3. Self-Supervised Hamiltonian Neural Networks --- Mattheakis, Sondak, Dogra & Protopapas, 2022

**3.1 Core Paradigm & Objective**

This paper takes a fundamentally different approach from data-driven HNNs. Instead of *learning* an unknown Hamiltonian from trajectory data, the Hamiltonian $H(q,p)$ is *assumed known*, and the neural network is used as a *solver* for Hamilton's equations of motion. This is an equation-driven (self-supervised), fully data-free method. The network discovers phase-space trajectories $\hat{z}(t)$ that satisfy the known Hamilton's equations over a continuous time domain, without any ground-truth trajectory data. The key advantage over traditional symplectic integrators is that the NN evaluates each time point independently (no sequential error accumulation), conserves the *original* Hamiltonian (not a perturbed one), and admits efficient GPU parallelization.

**3.2 Mathematical Formulation**

*Mathematical Model:* Parametric trial solutions enforce initial conditions exactly:

$$\hat{z}(t) = z(0) + f(t) \cdot N(t; \theta)$$

where $N(t;\theta) \in \mathbb{R}^{2d}$ is the NN output, $z(0)$ is the known initial state, and $f(t) = 1 - e^{-t}$ is a bounded parametric function satisfying $f(0) = 0$ (a crucial improvement over $f(t)=t$ which is unbounded for large $t$).

*Optimization Objective:* Minimize residuals of Hamilton's equations over $K$ collocation time points in $[0, T]$.

*Loss Function:*

$$\mathcal{L} = \frac{1}{K} \sum_{n=1}^{K} \left\| \dot{\hat{z}}^{(n)} - J \cdot \nabla_{\hat{z}^{(n)}} H(\hat{z}^{(n)}) \right\|^2 + \lambda \mathcal{L}_{\text{reg}}$$

$$\mathcal{L}_{\text{reg}} = \frac{1}{K} \sum_{n=1}^{K} \left( H(\hat{z}^{(n)}) - E_0 \right)^2$$

The regularization term penalizes deviations of the conserved energy from its known initial value $E_0$, stabilizing long-time predictions.

*Error bound:* The authors prove that the maximum component-wise error satisfies $\|{\delta z_i}\| \leq \ell_{\max} / \sigma_{\min}$, where $\ell_{\max}$ is the maximum instantaneous loss and $\sigma_{\min}$ is the minimum singular value of the Hessian of $H$. This allows pre-specifying the desired accuracy before training.

**3.3 Data & Training Requirements**

**Purely data-free.** No trajectory data is needed. The only inputs are (a) the analytic form of $H(q,p)$, (b) the initial condition $z(0)$, and (c) a set of $K$ collocation time points in $[0, T]$, randomly perturbed each epoch (acting as stochastic mini-batches). Uses sin activation functions for periodicity bias.

**3.4 Numerical Solver Interaction**

No numerical integrator is used at any point---neither in training nor inference. The trained network *is* the solver: it directly maps any time point $t$ to the approximate solution $\hat{z}(t)$. The method is compared against the symplectic Euler integrator, showing that the NN requires two orders of magnitude fewer evaluation points for equivalent accuracy.

---

### 4. Symplectic HNNs (SHNN) --- David & Mehats, 2023

**4.1 Core Paradigm & Objective**

This paper identifies a fundamental flaw in the original HNN training procedure: the loss function (Eq. 5/6 in their paper) implicitly performs a *forward Euler integration* step when comparing predicted and observed states. Forward Euler is *not symplectic*, meaning there exists *no* Hamiltonian function that the HNN could learn such that the loss is identically zero---an artificial lower bound is imposed on the training loss. SHNNs replace the forward Euler step with a *symplectic integration scheme* (symplectic Euler or implicit midpoint rule) inside the loss function, eliminating this lower bound and enabling the network to learn an exact *modified Hamiltonian* to arbitrary precision.

**4.2 Mathematical Formulation**

*Mathematical Model:* The loss is generalized using an abstract scheme function $s(y_0, y_1)$:

$$\mathcal{L}_{\text{SHNN}} = \left\| \frac{y_1 - y_0}{h} - J^{-1} \nabla \tilde{H}(s(y_0, y_1)) \right\|^2$$

For the symplectic Euler method: $s(y_0, y_1) = (p_1, q_0)$. For the implicit midpoint rule: $s(y_0, y_1) = (y_0 + y_1)/2$.

*Key Theorem (Proposition 1):* For the symplectic Euler and implicit midpoint schemes specifically (with general symplectic methods covered by B-series theory via Chartier et al.), there exists an exact smooth function $\tilde{H}$ (the *modified Hamiltonian*) such that the loss can be driven to zero. This function satisfies $\tilde{H} = H + \mathcal{O}(h^r)$ where $r$ is the order of the symplectic method used.

*Post-training correction:* Using backward error analysis and the Hamilton-Jacobi PDE, the true Hamiltonian can be recovered from the learned modified Hamiltonian. For symplectic Euler:

$$H = \tilde{H} - \frac{h}{2} \nabla_p \tilde{H} \cdot \nabla_q \tilde{H} + \mathcal{O}(h^2)$$

For implicit midpoint:

$$H = \tilde{H} - \frac{h^2}{24} \nabla^2 \tilde{H}(J^{-1}\nabla\tilde{H}, J^{-1}\nabla\tilde{H}) + \mathcal{O}(h^4)$$

*Loss Function:* Same structure as the SHNN loss above, with the scheme function $s$ chosen according to the desired symplectic method.

**4.3 Data & Training Requirements**

Requires discrete state-to-state transition pairs $(y_0, y_1 = \varphi_h(y_0))$ separated by a fixed time step $h$. Crucially, *no explicit time-derivative labels* $\dot{y}$ are needed. Data points $y_0$ are sampled uniformly from a bounded region of phase space with $y_1$ obtained from the true flow.

**4.4 Numerical Solver Interaction**

The symplectic integrator is *embedded within the loss function itself* during training, but only implicitly---since both $y_0$ and $y_1$ are known data, no iterative solve is needed (implicit methods have the same cost as explicit ones during training). At inference, the learned modified Hamiltonian is integrated with a standard integrator (Dormand-Prince RK5(4)). A key subtlety: integrating $\tilde{H}$ with the *same* symplectic method and *same* step size $h$ used in training recovers the *true* flow exactly.

---

### 5. Adaptable / Parameter-Cognizant HNNs --- Han, Glaz, Haile & Lai, 2021

**5.1 Core Paradigm & Objective**

Previous HNNs are trained and tested at the *same* bifurcation parameter values. This paper addresses *adaptability*: can an HNN trained on time series from a small number of parameter values predict dynamics at *unseen* parameter values? The answer is achieved by adding a dedicated *parameter input channel* $\alpha$ to the HNN architecture, making the network cognizant of how dynamics change with system parameters. This enables prediction of routes to chaos, bifurcation diagrams, and transitions between integrable and chaotic regimes.

**5.2 Mathematical Formulation**

*Mathematical Model:* The HNN learns $H_\theta(q, p, \alpha)$ where $\alpha$ is the bifurcation parameter of the target system. The network architecture is a standard feed-forward HNN (2 hidden layers, 200 neurons each) with an additional input channel for $\alpha$, kept structurally separate from $(q,p)$ so that the symplectic structure of Hamilton's equations is preserved:

$$\dot{q} = \frac{\partial H_\theta}{\partial p}, \quad \dot{p} = -\frac{\partial H_\theta}{\partial q}$$

The parameter $\alpha$ is *not* differentiated---it acts as a conditioning variable.

*Loss Function:*

$$\mathcal{L} = \sum_i \left| \frac{\partial H}{\partial q} + \frac{dp_{\text{real}}}{dt} \right| + \left| \frac{\partial H}{\partial p} - \frac{dq_{\text{real}}}{dt} \right|$$

Standard HNN loss, computed across multiple parameter values simultaneously.

**5.3 Data & Training Requirements**

Requires time-derivative data $(\dot{q}, \dot{p})$ at multiple parameter values. Training uses as few as 4 distinct parameter values; the network then interpolates/extrapolates across a continuous parameter interval. Demonstrated on the Henon-Heiles system and Morse potential.

**5.4 Numerical Solver Interaction**

External RK4 integration at inference. The integrator is not embedded in the training loop. Predictions are validated via the ensemble maximum Lyapunov exponent and alignment index to assess whether the learned dynamics correctly reproduce chaos transitions.

---

### 6. Hamiltonian Generative Networks (HGN) --- Toth, Rezende, Racaniere, Botev, Jaegle & Higgins, 2020

**6.1 Core Paradigm & Objective**

HGN addresses the observation-to-state problem that the original HNN only partially tackled: learning Hamiltonian dynamics directly from *high-dimensional pixel observations* without restrictive domain assumptions. While Greydanus et al. required the latent space dimensionality to match the true phase space and assumed $p \approx \dot{q}$, HGN uses a VAE-based inference network to learn an abstract phase space of *arbitrary* dimensionality, rolls out dynamics using a learned Hamiltonian with a leapfrog integrator, and reconstructs images from the position part of the latent state.

**6.2 Mathematical Formulation**

*Mathematical Model:* Three-component architecture:

1. **Inference network** (encoder): Takes a stacked sequence of images $(x_0, \ldots, x_T)$ and outputs a posterior $q_\phi(z | x_0, \ldots, x_T)$ over the initial abstract state $z$.
2. **Hamiltonian network**: Maps $s_0 = f_\psi(z)$ (where the state is split into abstract $(q,p)$) through the learned Hamiltonian $H_\gamma(q,p)$ using leapfrog integration to produce a trajectory $s_0, s_1, \ldots, s_T$.
3. **Decoder**: Reconstructs images from position variables only: $\hat{x}_t = d_\theta(q_t)$.

*Loss Function (temporally-extended VAE):*

$$\mathcal{L} = \frac{1}{T+1} \sum_{t=0}^{T} \mathbb{E}_{q_\phi(z|x)}[\log p_{\psi,\gamma,\theta}(x_t | q_t)] - \text{KL}(q_\phi(z) \| p(z))$$

No direct supervision of Hamilton's equations---the Hamiltonian structure is enforced *architecturally* through the leapfrog rollout. The loss backpropagates through the integrator.

**6.3 Data & Training Requirements**

Requires sequences of high-dimensional observations (e.g., 28x28 or 64x64 pixel images) showing the temporal evolution of a physical system. No phase-space coordinates, no derivatives, no knowledge of the Hamiltonian or the system's dimensionality needed. Demonstrated on pendulum, mass-spring, 2-body, and 3-body systems.

**6.4 Numerical Solver Interaction**

A leapfrog (Stormer-Verlet) integrator is *embedded within the computation graph* during both training and inference. Gradients flow through the integrator during backpropagation. The separable Hamiltonian assumption enables the use of this explicit symplectic integrator.

---

### 7. Symplectic Neural Operators (SNO) --- Makara, Tanaka, Matsubara & Yaguchi, 2026

**7.1 Core Paradigm & Objective**

All previous HNN variants operate in *finite-dimensional* phase spaces ($\mathbb{R}^{2n}$). SNO extends structure-preserving learning to *infinite-dimensional* Hamiltonian systems (Hamiltonian PDEs). Standard neural operators (FNO, DeepONet, etc.) can learn PDE solution maps but do not preserve the symplectic structure, leading to energy drift and instability under long-time rollout. SNO constructs a neural operator architecture whose learned flow map is symplectic *by construction* in function space, combining the geometric guarantees of symplectic integration with the mesh-independent, function-to-function surrogate capability of neural operators.

**7.2 Mathematical Formulation**

*Mathematical Model:* The infinite-dimensional Hamiltonian system is:

$$\dot{u}(t) = J^{-1} \nabla H(u(t)), \quad u \in \mathcal{P} = \mathcal{H} \oplus \mathcal{H}$$

where $\mathcal{P}$ is a Hilbert phase space and $J$ is the canonical symplectic operator. SNO directly learns the time-$\tau$ flow map $\Phi^\tau_H: (q(\cdot), p(\cdot)) \mapsto (q(\cdot,\tau), p(\cdot,\tau))$, rather than learning the Hamiltonian functional itself.

The architecture is a composition of *symplectic shear operators*:

$$\Phi_V(q, p) = (q, \; p + \nabla V(q)), \quad \Psi_W(q, p) = (q + \nabla W(p), \; p)$$

where $V, W: \mathcal{H} \to \mathbb{R}$ are parameterized by *Self-Adjoint Fourier Neural Operators* (SAFNOs)---a new class of neural operators that enforce self-adjointness of the Hessian operator, which is necessary for symplecticity in infinite dimensions.

*Optimization Objective:* Empirical risk minimization on one-step pairs:

$$\theta^\star \in \arg\min_\theta \frac{1}{N} \sum_{i=1}^{N} \|N_\theta(u^{(i)}) - v^{(i)}\|^2_{\mathcal{P}}$$

where $v^{(i)} \approx \Phi^\tau(u^{(i)})$.

*Loss Function:* L2 loss (or Sobolev-type norms) between predicted and true next-state *functions*.

*Key Theorem:* If the learned map is symplectic and constitutes a smooth $\varepsilon$-perturbation of the exact flow (in the Hamiltonian sense), then it nearly preserves a modified Hamiltonian over long-time evolution---the infinite-dimensional analogue of backward error analysis.

**7.3 Data & Training Requirements**

Requires pairs of *functions* $(u_n(\cdot), u_{n+1}(\cdot))$ representing the state field at consecutive fixed time steps $\tau$. Discretized on spatial grids but the architecture generalizes across resolutions (mesh-independent). No explicit knowledge of the Hamiltonian functional is needed.

**7.4 Numerical Solver Interaction**

No external integrator is used. The SNO *is* the integrator---it directly maps functions at time $t$ to functions at time $t+\tau$. Long-time rollouts are performed by iterated composition $\hat{u}_{n+1} = N_\theta(\hat{u}_n)$, analogous to a time-stepper. The symplectic structure prevents catastrophic energy drift during these rollouts.

---

### 8. Stable Port-Hamiltonian Neural Networks (sPHNN) --- Roth, Klein, Kannapinn, Peters & Weeger, 2025

**8.1 Core Paradigm & Objective**

Previous port-Hamiltonian architectures (pHNNs, Desai et al.) can model dissipation and forcing but provide no guarantees on *stability*. The learned dynamics may exhibit unbounded trajectories, spurious equilibria, or sensitivity to perturbations---critical failure modes for engineering deployment. sPHNNs enforce *global Lyapunov stability* by construction: the Hamiltonian is parameterized as a *fully input-convex neural network* (FICNN) with a known strict minimum, guaranteeing that (a) all trajectories are bounded, and (b) the system converges to the unique equilibrium when dissipation is present ($R \succ 0$).

**8.2 Mathematical Formulation**

*Mathematical Model:* Full port-Hamiltonian ODE:

$$\dot{x} = [J(x) - R(x)] \frac{\partial H}{\partial x}(x) + G(x) u(t)$$

where $J(x) = -J^\top(x)$ (skew-symmetric structure matrix), $R(x) = R^\top(x) \succeq 0$ (positive semi-definite dissipation matrix via Cholesky $R = LL^\top$), $G(x)$ (input matrix), and $H(x)$ is the Hamiltonian.

*Stability Theorem (Theorem 3.1):* If $H$ is convex, twice continuously differentiable, with $H(0) = 0$, $\partial H / \partial x|_0 = 0$, and $\partial^2 H / \partial x^2|_0 \succ 0$, then $H$ is a valid Lyapunov function, all solutions are bounded, and the equilibrium is globally asymptotically stable when $R(x) \succ 0$.

*Hamiltonian parameterization:* FICNN $f(x)$ is normalized to enforce the minimum at a known equilibrium $x^*$:

$$H(x) = f(x) - f(x^*) - \left.\frac{\partial f}{\partial x}\right|_{x^*}^\top (x - x^*) + \varepsilon \|x - x^*\|^2$$

The convexity of $f$ (via non-negative weight constraints and convex non-decreasing activations like softplus) combined with this normalization ensures all conditions of Theorem 3.1.

*Loss Function:* Mean squared error---either derivative fitting ($\|\hat{\dot{x}} - \dot{x}\|^2$) or trajectory fitting (comparing integrated trajectories). Trajectory fitting backpropagates through the ODE solver.

**8.3 Data & Training Requirements**

Supports both state-derivative pairs $(\dot{x})$ for derivative fitting and discrete trajectory data for trajectory fitting. Demonstrated on measurement data from a real cascaded-tanks system (1024 data points from a single trajectory), multiphysics simulation data, and synthetic benchmarks. External input signals $u(t)$ are part of the data.

**8.4 Numerical Solver Interaction**

For trajectory fitting, an adaptive Runge-Kutta integrator (Tsit5) is embedded in the training loop, with gradients propagated through the solver (or via adjoint sensitivity). For derivative fitting, no integrator is needed during training. At inference, the same Tsit5 integrator is used with adaptive step sizes.

---

## PHASE 2: Cross-Referencing & Comparative Matrix

### Comparison Table

| Model Variant | System Type | Symplecticity | Dissipation/Control | Internal Solver Requirement | Critical Architectural Limitation |
|---|---|---|---|---|---|
| **HNN** (Greydanus 2019) | Conservative only | Preserved (by construction of symplectic gradient) | No | None (external RK45 at inference) | Cannot model friction, damping, or external forces. Requires explicit $\dot{x}$ labels. Energy drift from non-symplectic integration at inference. |
| **pHNN** (Desai 2021) | Non-conservative (forced, damped) | Not strictly preserved (energy dissipation is modeled) | Yes --- separate learnable $F(t)$ and $N$ | None (external RK4) or optional embedded RK4 | Identifiability challenge between $H$ and $F$ (information leakage). Requires $\lambda_F, \lambda_N$ hyperparameter tuning. Cannot guarantee stability. |
| **Self-Supervised HNN** (Mattheakis 2022) | Conservative (known $H$) | Preserved (NN satisfies Hamilton's eqs. to arbitrary precision; conserves *original* $H$) | No | None (the NN *is* the solver) | Requires full analytic knowledge of $H$---cannot discover unknown dynamics. Accuracy degrades outside training time interval $[0,T]$. |
| **SHNN** (David & Mehats 2023) | Conservative only | Strictly preserved (symplectic scheme in loss eliminates artificial lower bound) | No | Symplectic integrator embedded in loss function | Limited to conservative systems. Modified Hamiltonian differs from true $H$ by $\mathcal{O}(h^r)$; correction requires symbolic computation. |
| **Adaptable HNN** (Han 2021) | Conservative (parameter-varying) | Preserved (standard HNN structure) | No | None (external RK4) | Still limited to conservative systems. Prediction degrades as target parameter deviates far from training set. Requires $\dot{x}$ labels at multiple parameter values. |
| **HGN** (Toth 2020) | Conservative (from pixels) | Approximate (leapfrog integrator preserves symplecticity in abstract latent space) | No | Leapfrog integrator embedded in training loop | Assumes separable Hamiltonian for leapfrog. Latent space may not correspond to physical phase space. No guarantees on physical interpretability of learned $H$. |
| **SNO** (Makara 2026) | Conservative (infinite-dimensional / PDEs) | Strictly preserved (symplectic shear composition by construction) | No | None (SNO *is* the time-stepper) | Limited to conservative Hamiltonian PDEs. Requires self-adjointness of Hessian operators. Computational cost of SAFNOs. Fixed $\tau$ step---intermediate states inaccessible. |
| **sPHNN** (Roth 2025) | Non-conservative (forced, damped, stable) | Not preserved (energy dissipation by design) | Yes --- learnable $J(x)$, $R(x)$, $G(x)u(t)$ | Adaptive RK (Tsit5) embedded for trajectory fitting; none for derivative fitting | Convexity of $H$ (FICNN) limits expressiveness---cannot represent systems with multiple equilibria or non-convex energy landscapes. Single global attractor only. |

### Cross-Reference Lineage

**HNN (2019) $\to$ pHNN (2021):** pHNNs directly address the central limitation of HNNs---their inability to handle non-conservative systems. By embedding the port-Hamiltonian formalism, pHNNs decompose the dynamics into conservative ($H$), dissipative ($N$), and driven ($F(t)$) components. The L1 regularization on $F$ and $N$ ensures the model gracefully degrades to a standard HNN when the system is truly conservative.

**HNN (2019) $\to$ SHNN (2023):** David & Mehats identify that the HNN loss function implicitly uses forward Euler, which is non-symplectic. This introduces an irremovable lower bound on the training loss. SHNNs replace this with symplectic schemes (symplectic Euler, implicit midpoint), proving that an exact learnable function (the modified Hamiltonian) always exists. The post-training correction via backward error analysis then recovers the true $H$.

**HNN (2019) $\to$ HGN (2020):** HGN addresses the pixel-observation limitation. While Greydanus et al. attempted pixel learning via a constrained autoencoder (requiring known latent dimensionality and $p \approx \dot{q}$), HGN uses a full VAE with arbitrary latent dimensionality and no assumptions on the form of momenta. The leapfrog integrator replaces the Euler-based approach, providing better symplectic structure in the latent rollout.

**HNN (2019) $\to$ Adaptable HNN (2021):** Han et al. extend HNNs from single-parameter-value training to cross-parameter prediction by adding a parameter input channel. The key insight is that the parameter must be kept structurally separate from $(q,p)$ to preserve the symplectic gradient structure.

**Self-Supervised HNN (2022) vs. Data-Driven HNN (2019):** These are conceptually *dual* problems. HNN learns an unknown $H$ from trajectory data; Self-Supervised HNN takes a known $H$ and discovers trajectories. Both exploit the Hamiltonian structure but for opposite directions of the inference problem.

**pHNN (2021) $\to$ sPHNN (2025):** sPHNNs correct two critical gaps in pHNNs: (1) the lack of stability guarantees (pHNN can learn dynamics with unbounded trajectories or spurious attractors), and (2) the limited expressiveness of the dissipation model (pHNN uses a scalar $N$; sPHNN uses a full state-dependent matrix $R(x)$ via Cholesky factorization). The FICNN Hamiltonian and Lyapunov stability theorem provide formal guarantees absent in pHNNs.

**Finite-dimensional HNNs $\to$ SNO (2026):** SNO extends the HNN paradigm from $\mathbb{R}^{2n}$ to function spaces, addressing Hamiltonian PDEs. The core architectural trick---composing symplectic shear maps---is the infinite-dimensional analogue of SympNets (Jin et al., 2020) used in finite dimensions. The SAFNO construction solves a non-trivial technical challenge: enforcing self-adjointness of operators on function spaces (trivial symmetrization $A \to (A+A^\top)/2$ does not generalize to infinite dimensions).

---

## PHASE 3: Pattern Matching & Trade-Off Synthesis

### 1. Architectural Evolution

The trajectory of HNN development follows a clear progression through four stages:

**Stage 1 --- Idealized Conservation (2019):** The original HNN establishes the core insight: parameterize the *scalar* Hamiltonian, not the *vector field*, and derive dynamics via the symplectic gradient. This guarantees exact conservation of a learned energy-like quantity. The architecture is minimal (a single MLP) and the scope is narrow (conservative, autonomous, low-dimensional, phase-space data with $\dot{x}$ labels).

**Stage 2 --- Relaxing Physical Assumptions (2020--2021):** Multiple papers independently extend the formalism to handle realistic physics. pHNNs add dissipation and forcing via the port-Hamiltonian framework. HGNs lift the requirement for phase-space data by learning from pixels. Adaptable HNNs add parameter-varying dynamics. Each extension preserves the core *scalar-function-then-differentiate* paradigm while relaxing one constraint.

**Stage 3 --- Numerical Rigor and Guarantees (2022--2023):** The focus shifts from *what physics can be modeled* to *how well*. SHNNs formally analyze the numerical integration scheme implicit in the loss function and prove that symplectic training enables exact learning of a modified Hamiltonian, with provable error bounds. Self-Supervised HNNs prove error bounds relating network performance to trajectory accuracy. The emphasis is on explainability and mathematical guarantees, not just empirical performance.

**Stage 4 --- Engineering Deployment (2025--2026):** The latest models address the gap between academic benchmarks and real-world engineering use. sPHNNs guarantee global Lyapunov stability---essential for safety-critical applications like robotics and control. SNOs extend the paradigm to infinite-dimensional systems (PDEs), enabling structure-preserving surrogates for computational physics. The inductive biases are now strong enough to learn from sparse, noisy, real-world measurement data (e.g., sPHNN on the cascaded tanks dataset).

### 2. The Data vs. Prior Knowledge Trade-Off

There is a clear inverse relationship between the amount of physical structure injected into the architecture and the data requirements:

**Maximum data, minimum prior:** The original HNN and baseline NNs require dense phase-space trajectories with explicit $\dot{x}$ labels and no assumption about the system's structure beyond "it's Hamiltonian." The architecture provides the weakest bias (just the symplectic gradient) and needs the most data.

**Less data via structural decomposition:** pHNNs and sPHNNs encode the *form* of the dynamics ($J$, $R$, $G$ structure) and separate conservative from dissipative contributions. This stronger bias allows learning from sparser data and shorter trajectories. The sPHNN cascaded-tanks example uses a *single* trajectory of 1024 points.

**Eliminating $\dot{x}$ labels:** SHNNs achieve this by embedding a symplectic integrator in the loss, requiring only discrete state pairs $(y_0, y_1)$ rather than continuous derivative data. Similarly, HGNs and trajectory-fitting sPHNNs avoid $\dot{x}$ labels entirely by backpropagating through an integrator.

**Zero data (maximum prior):** Self-Supervised HNNs represent the extreme---when the full Hamiltonian is analytically known, no trajectory data is needed at all. The NN discovers solutions purely from the differential equation structure and initial conditions.

**Cross-resolution generalization:** SNOs push this further by encoding symplecticity at the operator level, enabling mesh-independent generalization. The architectural prior (symplectic shear composition) is so strong that the model transfers across different spatial discretizations.

### 3. Emerging Patterns

Several mathematical tricks and inductive biases recur across multiple independent papers:

**Symplectic gradient as universal structural prior:** Every paper in this survey, without exception, uses the key insight of Greydanus et al.: learn a scalar function $H$ and derive the vector field via $\dot{z} = J^{-1}\nabla H$. This single architectural choice eliminates $2n-1$ degrees of freedom in the vector field (reducing it to the gradient of one scalar) and enforces conservation by construction.

**Backward error analysis / modified Hamiltonians:** Both SHNNs and SNOs use the same deep result from geometric numerical integration: a symplectic map is *exactly* the flow of a nearby (modified) Hamiltonian. This provides both explainability (we know *what* the network learned) and a path to correction (series inversion to recover the true $H$).

**Cholesky factorization for positive-definiteness:** Both pHNNs (implicitly, via the constraint $R \succeq 0$) and sPHNNs (explicitly, $R = LL^\top$) use Cholesky decomposition to enforce positive semi-definiteness of the dissipation matrix without constrained optimization. This is a general trick for enforcing matrix inequalities in differentiable architectures.

**Convexity-based stability:** sPHNNs use FICNNs to enforce convexity of $H$, making it a valid Lyapunov function. This mirrors the classical result in control theory (LaSalle's invariance principle) and provides deployment-grade stability guarantees. The pattern of using architecturally-constrained networks (non-negative weights, monotone activations) to enforce convexity is also seen in input-convex neural networks for optimal transport and other domains.

**Separation of concerns / additive decomposition:** Multiple papers decompose the dynamics into interpretable additive components: conservative + dissipative (pHNN, sPHNN), kinetic + potential (separable Hamiltonians in HGN), position-dependent + momentum-dependent shears (SNO, SympNets). This modular approach improves both interpretability and learnability.

**Energy regularization for long-time stability:** Both Self-Supervised HNNs (energy deviation penalty $\mathcal{L}_{\text{reg}}$) and the general HNN literature use energy-based regularization to prevent long-time drift. This is the neural analogue of the classical principle that symplectic integrators conserve a modified energy---the regularization explicitly encourages conservation of the *original* energy.

**Trigonometric activation functions:** Both Self-Supervised HNNs and DiPietro et al. (cited in the review) use $\sin(\cdot)$ activations, exploiting the periodicity bias inherent to oscillatory Hamiltonian systems. This consistently outperforms sigmoid/ReLU for dynamical systems with periodic or quasi-periodic orbits.
