# SuGaR Geometry Integration Notes

This project uses SuGaR as a geometry prior for semantic grouping, not as a mesh
reconstruction pipeline. The integration deliberately keeps the successful
lightweight SuGaR v1 behavior and avoids routing SuGaR surface scores directly
into the semantic contrastive weights.

## Paper-to-Code Mapping

1. SuGaR paper Section 4.1, "Regularization for Surface-Aligned 3D Gaussians"

   SuGaR's core idea is that a Gaussian can represent a small local surface
   element when its smallest scaling axis is aligned with the local surface
   normal and the Gaussian is flattened along that axis. In this codebase this
   maps to `loss_sugar_surface_alignment` in `utils/loss_utils.py`.

   The implementation uses a KNN neighborhood over Gaussian centers, estimates a
   local PCA normal from detached neighbors, and penalizes:

   - mismatch between the smallest Gaussian axis and the PCA normal
   - point-to-local-plane residual around the Gaussian
   - insufficient flattening of the smallest scale relative to tangent scales

2. SuGaR code structure, `sugar_scene/sugar_model.py`

   SuGaR exposes geometry accessors such as the smallest axis / normals so later
   stages can treat Gaussians as oriented surface elements. This project now
   mirrors that structure through `GaussianModel.get_surface_axis`,
   `GaussianModel.get_surface_thickness`, and
   `GaussianModel.get_surface_flat_ratio`.

   These accessors are intentionally diagnostic and reusable. They do not change
   semantic grouping by themselves, so they are safe to use for logging,
   visualization, or future ablations.

3. SuGaR training code, coarse regularization and opacity entropy

   SuGaR uses opacity-related regularization in its surface reconstruction
   schedule. For semantic grouping, hard opacity pruning is risky because it can
   remove Gaussians that still carry useful object features. This project adds
   `loss_sugar_opacity_entropy` as an optional, disabled-by-default regularizer.

   To test it, set `sugar_lambda_opacity_entropy` to a small value such as
   `0.002` or `0.005` during the short geometry warmup window.

## What Is Not Ported

SuGaR Sections 4.2 and 4.3 focus on extracting a mesh with Poisson
reconstruction and binding Gaussians to mesh triangles. That is not directly
used here because the target output is a semantic Gaussian grouping, not an
editable mesh. Pulling mesh extraction into the training loop would add a heavy
dependency and does not directly improve 3D semantic labels.

## Current Safe Default

The current configuration keeps the proven lightweight SuGaR v1 behavior:

- `sugar_weight_lambda = 0.2`
- `sugar_lambda_axis = 1.0`
- `sugar_lambda_plane = 0.5`
- `sugar_lambda_flat = 0.1`
- `sugar_lambda_opacity_entropy = 0.0`

The recommended next ablation is to first reproduce the v1 result with opacity
entropy disabled, then try only one small positive entropy weight in the
`[sugar_start_iter, geo_start_iter)` window.
