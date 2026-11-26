# Third-party notices

## DiffusionDriveV2

The integration targets [DiffusionDriveV2](https://github.com/hustvl/DiffusionDriveV2)
at commit `1cd12a1e155c34dcc471261835444c8d5587580b`.
Its LICENSE file specifies the MIT License, Copyright (c) 2024 HUST Vision Lab.
The full notice is retained in `licenses/DiffusionDriveV2-MIT.txt`.

`patches/energy_reward.patch` modifies the RL configuration and model to expose
the candidate-energy reward branch and final-candidate capture. The default
branch retains the upstream reward. The patch includes upstream context lines;
their original copyright and license apply.

The project-specific code supplies road clearance, a dual-graph interaction
evaluator, candidate supervision, reward construction and training utilities.
The upstream image backbone, anchors, diffusion decoder and inference selector
are not reimplemented here.

## NAVSIM and nuPlan

The upstream framework incorporates NAVSIM. NAVSIM code is distributed under
Apache-2.0 unless a file specifies otherwise; retain its existing notices in
the upstream checkout. nuPlan-devkit and other dependencies retain their own
licenses. Installing this package does not replace those notices or license
the underlying datasets.

OpenScene logs, sensor files and nuPlan maps must be obtained through their
official access routes under the respective data terms. No dataset files,
annotations, candidate banks or trained weights are included here.

Sources:

- https://github.com/hustvl/DiffusionDriveV2/blob/1cd12a1e155c34dcc471261835444c8d5587580b/LICENSE
- https://github.com/autonomousvision/navsim/blob/v1.1/LICENSE
- https://github.com/motional/nuplan-devkit/tree/nuplan-devkit-v1.2
