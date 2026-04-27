import typing as tp

from torch import nn

from voicesdk.nn.feat import features, features_tf

from .resnettf import SpectralFeaturesTF

FEATS = tp.Literal[
    "pt",
    "pt_mel",
    "tf",
    "tf_mel",
    "tf_spec",
    "pt_stft"
]


class FeaturedModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        feat_type: tp.Optional[str],
        loss: tp.Optional[nn.Module] = None,
        params: dict[str, tp.Any] = {},
        hop_length: int | None = None,
        num_bins: int | None = None
    ):
        super().__init__()

        self._dim = None
        self._backbone = backbone
        self._loss = loss

        if num_bins is not None:
            print(f"num bins set to: {num_bins}")
            params["n_mels"] = num_bins
        if hop_length is not None:
            print(f"hop length set to: {hop_length}")
            params["hop_length"] = hop_length

        if feat_type is None:
            self.feat = None
        elif feat_type in ["pt", "pt_mel"]:
            self.feat = features.MelBanks(**params)
        elif feat_type in ["tf", "tf_mel"]:
            self.feat = features_tf.TFMelBanks(**params)
        elif feat_type == "tf_spec":
            self.feat = features_tf.TFSpectrogram(**params)
        elif feat_type == "pt_stft":
            self.feat = features.STFT(**params)
        elif feat_type == "spectral_feature_tf":
            self.feat = SpectralFeaturesTF(**params)

    def forward(self, x, y = None):
        if self.feat is not None:
            x = self.feat(x)

        x = self._backbone(x)

        if self._loss is not None:
            x = self._loss(x, y)

        return x
