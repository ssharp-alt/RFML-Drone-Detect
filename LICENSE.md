Copyright (c) 2026 [YOUR LLC LEGAL NAME HERE]. All rights reserved.

This repository (source code, documentation, configuration, and any
associated trained model artifacts, whether included here or distributed
separately) is made available for VIEWING PURPOSES ONLY.

No permission is granted to use, copy, modify, merge, publish, distribute,
sublicense, deploy, host as a service, or sell copies of this software, in
whole or in part, for any purpose, without a separate written license
agreement with [YOUR LLC LEGAL NAME HERE].

The served model enforces this technically via a node-locked license
mechanism (see `license/`) -- the software will not run at all without a
valid, signed license issued for the specific machine it is deployed on.

To request a commercial license or a free evaluation/trial license (for
integration testing before purchase), contact: [YOUR CONTACT EMAIL HERE].

------------------------------------------------------------------------

## Third-party data attribution

The model included/producible from this repository was trained using the
**"Noisy Drone RF Signal Classification v2"** dataset, created by sgluege
(https://www.kaggle.com/sgluege) and published at
https://www.kaggle.com/datasets/sgluege/noisy-drone-rf-signal-classification-v2,
licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**:
https://creativecommons.org/licenses/by/4.0/

No modified or unmodified copy of that dataset is redistributed as part of
this repository -- `src/train.py` streams individual samples from the
original source (a Kaggle-hosted archive, or an equivalent private mirror)
at training time. This attribution is provided in compliance with the CC
BY 4.0 license's attribution requirement for the derivative work (the
trained model) produced from that dataset.
