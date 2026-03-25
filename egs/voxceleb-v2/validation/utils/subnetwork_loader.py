import json
import typing as tp

import torch
import torch.nn as nn
import yaml

from voicesdk.nn.arch import ResNetTF, ResNetTFSubNetS


class SubnetworkLoader:
    """Loads a backbone + subnet head with optional weight initialization."""

    def __init__(
        self,
        *,
        # Backbone (teacher) args
        teacher_cfg_path: str,
        teacher_ckpt_path: tp.Optional[str] = None,
        # Subnet head args
        subnet_config_path: str,
        subnet_ckpt_path: str,
        subnet_ckpt_prefix: str = "cls_head",
        # Inference shape for dim probing: (batch, channels, length)
        probe_shape: tp.Tuple[int, ...] = (2, 1, 16000),
    ) -> None:
        self._teacher_cfg_path = teacher_cfg_path
        self._teacher_ckpt_path = teacher_ckpt_path
        self._subnet_config_path = subnet_config_path
        self._subnet_ckpt_path = subnet_ckpt_path
        self._subnet_ckpt_prefix = subnet_ckpt_prefix
        self._probe_shape = probe_shape

        self._model: nn.Module = self._build()
        self._dim: int = self._probe_output_dim()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def dim(self) -> int:
        return self._dim

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_yaml(self, path: str) -> dict:
        with open(path, "r") as fh:
            return dict(yaml.load(fh, Loader=yaml.FullLoader))

    def _build_backbone(self) -> nn.Module:
        cfg = self._load_yaml(self._teacher_cfg_path)
        backbone = ResNetTF(**cfg["model_args"])
        backbone.eval()

        if self._teacher_ckpt_path is not None:
            state_dict = torch.load(
                self._teacher_ckpt_path, map_location=torch.device("cpu")
            )
            backbone.load_state_dict(state_dict, strict=True)

        return backbone

    def _load_subnet_config(self) -> dict:
        with open(self._subnet_config_path, "r") as fh:
            raw = json.load(fh)

        cfg: dict = raw["model_config"]["cls_head"]["params"]
        cfg.pop("backbone_exp_dir_ep", None)
        cfg["subnets_configs"]["vcd"].pop("head_config", None)
        return cfg

    def _build(self) -> nn.Module:
        cfg = self._load_subnet_config()
        cfg["backbone"] = self._build_backbone()

        model = ResNetTFSubNetS(**cfg)
        self._load_subnet_weights(model)
        model.eval()
        return model

    def _load_subnet_weights(self, model: nn.Module) -> None:
        checkpoint = torch.load(
            self._subnet_ckpt_path, map_location=torch.device("cpu")
        )
        prefix = self._subnet_ckpt_prefix + "." if self._subnet_ckpt_prefix else ""

        state_dict = model.state_dict()
        for key, param in state_dict.items():
            ckpt_key = f"{prefix}{key}"
            if ckpt_key not in checkpoint:
                raise KeyError(
                    f"Key '{ckpt_key}' not found in checkpoint. "
                    f"Check subnet_ckpt_prefix='{self._subnet_ckpt_prefix}'."
                )
            param.copy_(checkpoint[ckpt_key])

    def _probe_output_dim(self) -> int:
        dummy = torch.randn(*self._probe_shape)
        with torch.no_grad():
            output = self._model(dummy)
        return int(output.shape[-1])
