# Module1 / Module2 Decomposition

This note freezes the clean ablation entry points for the current best stable research line.
It intentionally does not include Module3 optimization, true split, split-lite, boundary-reg,
candidate-safe update, local-anchor refinement, or multi-view reassociation.

## Goal

The next useful question is not whether a larger combined system can be tuned into a good
number. The useful question is whether Module1 and Module2 each have a stable marginal
contribution, and whether their combination is genuinely complementary.

## Current Reading Of Past Experiments

- Module1 is the most direct fit for the core failure mode: unreliable semantic propagation
  near geometric or semantic boundaries.
- Module2 has a complete implementation, but the measured gains are small and not yet
  stable across scenes. The current sweet spot is around `proto_lambda_cons = 0.20`.
- Stronger prototype consistency such as `proto_lambda_cons = 0.30` hurt ramen in earlier
  runs, so the clean Module2 line should stay at `cons020` first.
- Module3 split probe was useful as a diagnostic, but true split, split-lite, boundary-reg,
  and local anchor refinement all degraded the main metrics. Those should stay out of the
  next clean ablation.

## Config Entry Points

| Purpose | Config |
| --- | --- |
| Baseline without Module1 / Module2 | `config/gaussian_dataset/train_ablate_baseline_no_m1_m2.json` |
| Module1 only | `config/gaussian_dataset/train_ablate_module1_graph_only.json` |
| Module2 only, graph loss disabled | `config/gaussian_dataset/train_ablate_module2_proto_only_cons020.json` |
| Module1 + Module2 | `config/gaussian_dataset/train_ablate_module1_module2_cons020.json` |

## Important Interpretation Detail

`train_ablate_module2_proto_only_cons020.json` disables the graph contrastive loss by setting
`graph_start_iter` far beyond training and `graph_weight_lambda` to `0.0`.

The prototype loss still needs `graph_data` as local support for reliability filtering,
positive-edge assignment consistency, and diagnostics. Therefore this config means:

`Module2 optimization only, with graph context but without Module1 graph contrastive gradients.`

It should not be described as "no graph is computed."

## Recommended Commands

Run ramen first because earlier experiments showed it is more sensitive and less forgiving.

```bash
bash script/my_train.sh lerf/ramen 1 -m lerf/ramen_ablate_baseline_no_m1_m2 --config_file config/gaussian_dataset/train_ablate_baseline_no_m1_m2.json --eval
bash script/my_train.sh lerf/ramen 1 -m lerf/ramen_ablate_module1_graph_only --config_file config/gaussian_dataset/train_ablate_module1_graph_only.json --eval
bash script/my_train.sh lerf/ramen 1 -m lerf/ramen_ablate_module2_proto_only_cons020 --config_file config/gaussian_dataset/train_ablate_module2_proto_only_cons020.json --eval
bash script/my_train.sh lerf/ramen 1 -m lerf/ramen_ablate_module1_module2_cons020 --config_file config/gaussian_dataset/train_ablate_module1_module2_cons020.json --eval
```

Then repeat only the best two or three variants on figurines.

```bash
bash script/my_train.sh lerf/figurines 1 -m lerf/figurines_ablate_module1_graph_only --config_file config/gaussian_dataset/train_ablate_module1_graph_only.json --eval
bash script/my_train.sh lerf/figurines 1 -m lerf/figurines_ablate_module2_proto_only_cons020 --config_file config/gaussian_dataset/train_ablate_module2_proto_only_cons020.json --eval
bash script/my_train.sh lerf/figurines 1 -m lerf/figurines_ablate_module1_module2_cons020 --config_file config/gaussian_dataset/train_ablate_module1_module2_cons020.json --eval
```

## Decision Rules

- If Module1 improves Boundary IoU without hurting mIoU, keep it as the first main method
  component.
- If Module2 only is flat or negative, do not tune complex prototype tricks yet. First
  check whether Module2 helps only after Module1 has stabilized the feature field.
- If Module1 + Module2 is better than both single-module variants, keep the combined story.
- If Module1 + Module2 is worse than Module1 only, use Module1 as the main route and demote
  Module2 to an optional or follow-up route.

## Metrics To Record

- Overall Mean IoU.
- Overall Boundary Mean IoU.
- `loss_graph`, `graph_pos_ratio`, `graph_neg_ratio`, `graph_ignore_ratio`.
- `proto_update_selected_ratio`, `proto_update_usage_entropy`, `proto_update_usage_max`.
- `proto_pair_cosine_max`, `proto_pair_cosine_p90`.
- `proto_margin_p50`, `proto_assign_conf_p50`.

## Explicit Exclusions

- No true split.
- No new Gaussian primitives.
- No parent deletion.
- No reg3d replacement.
- No boundary-reg.
- No candidate hard negative contrast.
- No local anchor contrast.
- No prototype EMA change.
