# Superseded: the laptop's 150-epoch CPU run

These are the first models trained on the full 10,987-crystal set, on the
laptop's CPU: 150 epochs with the plateau LR schedule, ~2 hours each.

They were replaced by the Colab GPU run, which trained the same architecture
for 200 epochs with cosine annealing in ~7 minutes per model, and did slightly
better (K_VRH test MAE 0.0696 vs 0.0708). More importantly the GPU run produced
three members per target, so it supports an ensemble; these two cannot.

Kept because they are the only models trained with the plateau schedule, so
they are the reference point for what the switch to cosine actually bought.
Both runs used --split-seed 42, so every number here is directly comparable to
the ones in results/.
