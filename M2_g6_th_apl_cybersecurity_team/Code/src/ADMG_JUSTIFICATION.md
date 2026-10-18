# Why an ADMG for microservice latency (written justification, stage 3)

Short version: a service's latency is influenced by things we do not measure, and a DAG over the measured services cannot represent that
without inventing wrong edges. An acyclic directed mixed graph (ADMG) is the exact marginal of a DAG with hidden nodes, so it is the
smallest graph class that stays correct when some nodes are unobserved. Below, what is argued from structure and what our own results
support or contradict.

## 1. The measured services are a marginal of a larger system
In PetShop the service map has 41-44 nodes per scenario, but 9-13 of them are not usable observed variables in our setting (no latency
column, mostly missing, or constant: e.g. managed queues, storage, step-function and notification services). In `pgm/petshop.py` they are
kept as *latent* nodes of a DAG over observed + latent, and the ground truth is the latent projection of that DAG (`pgm/graphs.py:
latent_projection`), then its maximal ancestral graph and PAG. So the benchmark's own truth is an ADMG/MAG, not a DAG.

## 2. What goes wrong with a DAG on the observed services
Latent projection (Verma-Pearl) turns a DAG over V u L into an ADMG over V with two edge types:
* `a -> b` when a reaches b through a path whose inner nodes are all latent (a latent middleman does not remove the effect, it hides the hop);
* `a <-> b` when a hidden node has directed paths to both (a shared, unmeasured cause: a common database, queue, host, network path or traffic surge).
A DAG learner must explain such a dependence by a directed edge, in one direction or the other, between services that do not influence each other.
Those edges are wrong in precisely the way root-cause analysis is sensitive to: a victim of a shared hidden cause would look like a cause or an
effect of its neighbour. The bidirected edge says "associated, neither causes the other", and an ancestral ADMG keeps the ancestral (who-can-affect-whom)
relations that a root-cause search needs.

## 3. Why the *ancestral* class and a linear-Gaussian parametrisation
* Ancestral ADMGs (no bidirected edge between a node and its ancestor) are the class whose Markov equivalence is represented by a MAG/PAG, the output
  of FCI-type methods and the object scored in the SPOT paper; ABIC and SPOT learn exactly this class with a differentiable constraint.
* Latency is roughly additive along a call chain (a caller waits for its callees), so a linear structural model is a natural first approximation,
  and the linear-Gaussian ADMG has a likelihood (RICF), a covariance `Sigma = (I-W1)^-T Omega (I-W1)^-1` and therefore an *inference* step:
  stage 3 uses the fitted `Omega` (bidirected structure) to attribute an abnormal shift to one service's own mechanism (`pgm/rca.py`).
  A PAG with circle marks has no such parametric model.

## 3b. What the inference step does with the bidirected part
Given the fitted normal-period model, the shift of the structural noise is `d_eps = (I-W1)^T dmu`; a service that only passes on an upstream anomaly has
`d_eps ~ 0`. The score of service i is the Mahalanobis-distance reduction from a single shift at i, `(Omega^-1 d_eps)_i^2 / (Omega^-1)_ii`. With `Omega`
diagonal (DAG) this is the standardised residual shift given the parents; with bidirected edges, `Omega^-1` discounts the part of a shift that is
explained by noise shared with a confounded neighbour. Running `dag` and `admg` on the same graph isolates the benefit of the bidirected edges.

## 4. What the evidence in this project says (both ways)
* **Supports the structural argument:** the ground-truth graph only exists as an ADMG because of the 9-13 latent nodes; a DAG over the observed nodes would not be
  a faithful description of the benchmark.
* **Weakens the practical argument on PetShop:** after projection the truth contains 0-1 bidirected edges per scenario (low 1, high 0, temporal 1). PetShop's
  call graph is nearly a latent-mediator structure, so it is a *weak test* of the confounding benefit. Whether `admg` beats `dag` for RCA here is an empirical
  question answered by `results/rca_summary.md`, not something the structure guarantees.
* **A limitation of the learners:** ABIC and SPOT return 17-41 bidirected edges and only 6-14 directed ones (truth: 30-38 directed). On z-scored/rank-transformed
  data the pseudo-variable loss prefers bidirected explanations (stage 2a/2c finding). Bidirected edges in the *learned* graphs are therefore mostly artefacts, and
  tail F1 is 0.06-0.11. Using them in `Omega` may hurt as well as help; this is why the truth-MAG row is reported next to the learned graphs.

## 5. Assumptions that are known to be violated
* Linear-Gaussian: latency is right-skewed (median column skewness 1.7-2.6) and 5-minute series are autocorrelated (median lag-1 0.16-0.31). We work in a
  log or z-scored space fitted on normal data, which reduces but does not remove this.
* Acyclicity: users retrying slow pages create feedback that a DAG/ADMG cannot represent (also noted by the PetShop authors).
* Identifiability: the data identify the graph only up to Markov equivalence; orientation is the weakest part of the learned graphs.
* Stationarity: in the temporal-traffic scenarios the load changes over the day, so an abnormal window can differ from the normal period for reasons unrelated to the fault.

None of these invalidates the choice of graph class; they bound how much weight a learned-graph RCA result can carry.
