# Method

## Candidates

The integration retains the upstream scene encoder, trajectory anchors,
coordinate normalization, diffusion decoder and sampling schedule. During
training, each anchor produces a group of candidates. The captured trajectory
is the terminal sampling state, not a substituted ground-truth path.

Candidate poses use the current ego rear axle as the origin, x forward and y
left. Positions are in meters and headings in radians. `trajectory.pose_traj`
prepends the observed initial state and estimates future interval speeds from
successive positions divided by 0.5 seconds. It does not independently generate
or integrate a speed profile.

## Energy evaluation

Road energy samples the full vehicle footprint against a map-derived signed
distance field. It penalizes negative clearance after accounting for grid
coverage. Interaction energy combines a corner-interaction graph with a
vehicle-safety graph and produces a nonnegative candidate score.

Graph supervision evaluates the candidate against future recorded objects.
The label takes the maximum object risk at each future step and averages over
the horizon. Graph training uses Huber loss. Log-disjoint validation selects
the checkpoint before policy optimization.

Current object annotations and maps are privileged training context. They are
not additional inference inputs. `contexts.py` keeps future-label fields out
of the reward-facing interface.

## Reward and update

The default bounded cost and shaped reward are:

```text
cost = 0.5 * E_road / (1 + E_road)
     + 0.5 * E_interaction / (1 + E_interaction)
reward = PDM - strength * cost
```

The energy branch normalizes rewards within each anchor group, using a standard-
deviation floor of 0.05. Safe candidates retain clipped positive advantages
up to 5; candidates failing NC or DAC receive a negative advantage that also
depends on the energy cost. This branch removes the upstream GT reward gate.
`original` preserves the upstream advantage branch. Setting strength to zero
bypasses the energy branch as well.

The graph is frozen. Costs and advantages are detached, and the policy is
updated through the upstream transition log-probability surrogate with its
imitation term. The implementation does not differentiate through the graph
into the trajectory, modify candidates after sampling or retrain the selector.
The energy strength is a reward coefficient, not an inference gradient step.

## Output

Policy export changes only the permitted denoising-policy tensors in a full
SEL checkpoint. Inference keeps the upstream candidate expansion, coarse
selection and final fine-scoring selection. The selected output contains
eight future rear-axle positions and headings over four seconds.

Motion energy remains disabled. The configuration rejects a nonzero motion
weight rather than silently treating an unsupported term as implemented.
