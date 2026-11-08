# Local patches to minLoRA

Upstream: https://github.com/cccntu/minLoRA @ d47a4c93df87540bb396172cfac1f3246e7db5f9 (MIT, see LICENSE).
Vendored without `.git` and notebooks.

1. `minlora/model.py`: added `SingLoRAParametrization` (single-matrix LoRA, arXiv:2507.05566) and
   `set_singlora_ramp(model, value)`; both exported from `minlora/__init__.py`.
