import json
import os
import struct
from dataclasses import make_dataclass
from typing import List, Optional, Tuple, Union

import torch
import torchaudio
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchaudio.compliance.kaldi import fbank

from .usad_modules import ConformerEncoder, lengths_to_padding_mask

MAX_MEL_LENGTH = 3000  # 30 seconds


@torch.no_grad()
def wav_to_fbank(
    wavs: torch.Tensor,
    mel_dim: int = 128,
    norm_mean: float = -4.268,
    norm_std: float = 4.569,
    wav_lengths: Optional[torch.Tensor] = None,
    sample_rate: int = 16000,
    return_lengths: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Convert waveform to fbank features.

    Args:
        wavs (torch.Tensor): (B, T_wav) waveform tensor.
        mel_dim (int, optional): mel dimension. Defaults to 128.
        norm_mean (float, optional): mean for normalization. Defaults to -4.268.
        norm_std (float, optional): std for normalization. Defaults to 4.569.
        wav_lengths (torch.Tensor, optional): (B,) valid waveform lengths before padding.
        sample_rate (int, optional): waveform sample rate. Defaults to 16000.
        return_lengths (bool, optional): return exact fbank lengths. Defaults to False.

    Returns:
        torch.Tensor: (B, T_mel, mel_dim) fbank features. If return_lengths is True,
        also returns a (B,) tensor with exact feature lengths before padding.
    """
    # ref: https://github.com/cwx-worst-one/EAT/tree/main/feature_extract
    feature_dtype = wavs.dtype if wavs.is_floating_point() else torch.float32
    wavs_float = wavs.to(torch.float32)

    if wav_lengths is None:
        wav_lengths = torch.full(
            (wavs.shape[0],),
            wavs.shape[1],
            dtype=torch.long,
            device=wavs.device,
        )
    else:
        wav_lengths = wav_lengths.to(device=wavs.device, dtype=torch.long)
        if wav_lengths.dim() != 1 or wav_lengths.shape[0] != wavs.shape[0]:
            raise ValueError(
                "wav_lengths must be a 1-D tensor with batch size elements."
            )
        if torch.any(wav_lengths <= 0).item():
            raise ValueError("All wav_lengths values must be positive.")
        if torch.any(wav_lengths > wavs.shape[1]).item():
            raise ValueError(
                "wav_lengths cannot exceed the padded waveform length."
            )

    feats = []
    feat_lengths = []
    for i, wav_length in enumerate(wav_lengths.detach().cpu().tolist()):
        # Trim padding before centering so batched padding cannot affect valid audio.
        wav = wavs_float[i, :wav_length]
        wav = wav - wav.mean(dim=-1, keepdim=True)
        feat = fbank(
            wav.unsqueeze(0),
            htk_compat=True,
            sample_frequency=sample_rate,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=mel_dim,
            dither=0.0,
            frame_shift=10,
        )
        feat = feat[: feat.shape[0] - feat.shape[0] % 2, :]
        feat = (feat - norm_mean) / (norm_std * 2)
        feats.append(feat.to(dtype=feature_dtype))
        feat_lengths.append(feat.shape[0])

    mels = pad_sequence(feats, batch_first=True, padding_value=0.0)
    mel_lengths = torch.tensor(
        feat_lengths, dtype=torch.long, device=wavs.device
    )

    if return_lengths:
        return mels, mel_lengths
    return mels


_SAFETENSORS_DTYPES = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16, "F64": torch.float64}


def _read_safetensors(path: str) -> dict:
    """Minimal ``.safetensors`` reader (8-byte header size + JSON header + raw tensor bytes).

    Avoids a dependency on the ``safetensors`` package; supports floating point tensors only.
    """
    with open(path, "rb") as f:
        (header_size,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_size))
        data = bytearray(f.read())
    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        dtype = _SAFETENSORS_DTYPES[info["dtype"]]
        tensor = torch.frombuffer(data, dtype=dtype, count=(end - start) // dtype.itemsize, offset=start)
        tensors[name] = tensor.reshape(info["shape"]).clone()
    return tensors


class UsadModel(nn.Module):
    def __init__(self, cfg) -> None:
        """Initialize the UsadModel.
        Args:
            cfg: Configuration object containing model parameters.
        """
        super().__init__()

        self.cfg = cfg
        self.encoder = ConformerEncoder(cfg)
        self.max_mel_length = MAX_MEL_LENGTH

    @property
    def sample_rate(self) -> int:
        return 16000  # Hz

    @property
    def encoder_frame_rate(self) -> int:
        return round(100 / self.cfg.conv_subsample_rate)  # Hz

    @property
    def mel_dim(self) -> int:
        return self.cfg.input_dim

    @property
    def encoder_dim(self) -> int:
        return self.cfg.encoder_dim

    @property
    def num_layers(self) -> int:
        return self.cfg.num_layers

    @property
    def scene_embedding_size(self) -> int:
        return self.cfg.encoder_dim * self.cfg.num_layers

    @property
    def timestamp_embedding_size(self) -> int:
        return self.cfg.encoder_dim * self.cfg.num_layers

    @property
    def device(self) -> torch.device:
        """Get the device on which the model is located."""
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def set_audio_chunk_size(self, seconds: float = 30.0) -> None:
        """Set the maximum chunk size for feature extraction.
        Args:
            seconds (float, optional): Chunk size in seconds. Defaults to 30.0.
        """
        assert (
            seconds >= 0.1
        ), f"Chunk size must be greater than 0.1s, got {seconds} seconds."
        self.max_mel_length = int(seconds * 100)  # 100 Hz frame rate

    def load_audio(
        self, audio_path: str, move_to_device: bool = True
    ) -> torch.Tensor:
        """Load audio file and return waveform tensor.
        Args:
            audio_path (str): Path to the audio file.
        Returns:
            torch.Tensor: Waveform tensor of shape (wav_len,).
        """

        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sr, self.sample_rate
            )
        if waveform.shape[0] > 1:
            # If stereo, convert to mono by averaging channels
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = waveform.squeeze(0)  # Remove channel dimension if mono
        if move_to_device:
            return waveform.to(
                self.device
            )  # Ensure tensor is on the same device
        return waveform

    def load_audio_batch(
        self, audio_paths: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        wav_list = []
        wav_lengths = []
        for path in audio_paths:
            wav = self.load_audio(path, move_to_device=False)
            wav_list.append(wav)
            wav_lengths.append(wav.shape[0])
        wavs = pad_sequence(wav_list, batch_first=True).to(self.device)
        wav_lengths = torch.tensor(
            wav_lengths, dtype=torch.long, device=self.device
        )
        return wavs, wav_lengths

    def forward(
        self,
        wavs: torch.Tensor,
        wav_lengths: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        target_layer: Optional[int] = None,
        norm_mean: float = -4.268,
        norm_std: float = 4.569,
    ) -> dict:
        """
        Args:
            wavs (torch.Tensor): (B, T_wav) waveform tensor.
            wav_lengths (torch.Tensor, optional): (B,) lengths of each waveform. Defaults to None.
            padding_mask (torch.Tensor, optional): (B, T_wav) padding mask for the waveforms.
                If wav_lengths is not provided, this is used to infer valid lengths.
            target_layer (int, optional): If specified, only return the output of the target layer. Defaults to None (return all layers).
            norm_mean (float, optional): Mean for normalization. Defaults to -4.268.
            norm_std (float, optional): Std for normalization. Defaults to 4.569.
        Returns:
            dict: A dictionary containing the following keys:
                - "x": (B, T_out, encoder_dim) output of the encoder
                - "x_lengths": (B,) valid output lengths after encoder subsampling
                - "x_padding_mask": (B, T_out) output padding mask, where padding is True
                - "mel": (B, T_mel, mel_dim) input mel features
                - "mel_lengths": (B,) valid mel lengths before encoder subsampling
                - "hidden_states": list of (B, T_out, encoder_dim) hidden states of each layer
                - "ffn": list of (B, T_out, encoder_dim) output of the feed-forward network of each layer
        """

        # Check types
        assert isinstance(wavs, torch.Tensor), "wavs must be a torch.Tensor"
        assert wavs.dim() == 2, "wavs must be of shape (batch_size, seq_len)"
        if wav_lengths is not None:
            assert isinstance(
                wav_lengths, torch.Tensor
            ), "wav_lengths must be a torch.Tensor"
            assert (
                wav_lengths.dim() == 1
            ), "wav_lengths must be of shape (batch_size,)"
            assert (
                wav_lengths.shape[0] == wavs.shape[0]
            ), "wav_lengths must have the same batch size as wavs"
        if padding_mask is not None:
            assert isinstance(
                padding_mask, torch.Tensor
            ), "padding_mask must be a torch.Tensor"
            assert (
                padding_mask.dim() == 2
            ), "padding_mask must be of shape (batch_size, seq_len)"
            assert (
                padding_mask.shape[0] == wavs.shape[0]
            ), "padding_mask must have the same batch size as wavs"
            assert (
                padding_mask.shape[1] == wavs.shape[1]
            ), "padding_mask must have the same seq_len as wavs"
            if wav_lengths is None:
                wav_lengths = (~padding_mask.to(torch.bool)).sum(dim=1)
        if target_layer is not None:
            assert isinstance(
                target_layer, int
            ), "target_layer must be an int or None"
            assert (
                1 <= target_layer <= self.cfg.num_layers
            ), f"target_layer must be between 1 and {self.cfg.num_layers}"

        mel, mel_lengths = wav_to_fbank(
            wavs,
            wav_lengths=wav_lengths,
            mel_dim=self.mel_dim,
            norm_mean=norm_mean,
            norm_std=norm_std,
            sample_rate=self.sample_rate,
            return_lengths=True,
        )

        dtype = self.dtype

        if mel.dtype != dtype:
            mel = mel.to(dtype)

        num_layers = min(
            self.cfg.num_layers,
            target_layer if target_layer is not None else self.cfg.num_layers,
        )

        if mel.shape[1] <= self.max_mel_length:
            # If the mel length is less than or equal to max_mel_length, we can process it in one go
            x, x_len, layer_results = self.encoder(
                inputs=mel,
                input_lengths=mel_lengths,
                return_hidden=True,
                target_layer=target_layer,
            )

            result = {
                "x": x,
                "x_lengths": x_len,
                "x_padding_mask": lengths_to_padding_mask(
                    x_len, max_len=x.size(1)
                ),
                "mel": mel,
                "mel_lengths": mel_lengths,
                "hidden_states": layer_results["hidden_states"],
                "ffn": layer_results["ffn_1"],
            }
            return result

        # If the mel length is greater than max_mel_length, we need to process it in chunks
        result = {
            "x": [],
            "x_lengths": [],
            "mel": mel,
            "mel_lengths": mel_lengths,
            "hidden_states": [[] for _ in range(num_layers)],
            "ffn": [[] for _ in range(num_layers)],
        }
        for i in range(0, mel.shape[1], self.max_mel_length):
            if mel.shape[1] - i < 10:
                break

            _mel = mel[:, i : i + self.max_mel_length]
            _mel_lengths = None
            if mel_lengths is not None:
                _mel_lengths = torch.clamp(
                    mel_lengths - i, min=0, max=self.max_mel_length
                )

            x, x_len, layer_results = self.encoder(
                inputs=_mel,
                input_lengths=_mel_lengths,
                return_hidden=True,
                target_layer=target_layer,
            )

            result["x"].append(x)
            result["x_lengths"].append(x_len)
            for j in range(num_layers):
                result["hidden_states"][j].append(
                    layer_results["hidden_states"][j]
                )
                result["ffn"][j].append(layer_results["ffn_1"][j])

        result["x"] = torch.cat(result["x"], dim=1)
        result["x_lengths"] = torch.stack(result["x_lengths"], dim=0).sum(
            dim=0
        )
        result["x_padding_mask"] = lengths_to_padding_mask(
            result["x_lengths"], max_len=result["x"].size(1)
        )
        for j in range(num_layers):
            result["hidden_states"][j] = torch.cat(
                result["hidden_states"][j], dim=1
            )
            result["ffn"][j] = torch.cat(result["ffn"][j], dim=1)

        return result

    @classmethod
    def load_from_fairseq_ckpt(cls, ckpt_path: str):
        checkpoint = torch.load(ckpt_path, weights_only=False)
        config = checkpoint["cfg"]["model"]
        config = make_dataclass("Config", config.keys())(**config)
        model = cls(config)
        state_dict = checkpoint["model"]
        for k in list(state_dict.keys()):
            if not k.startswith("encoder."):
                del state_dict[k]
        model.load_state_dict(state_dict, strict=True)
        return model

    # keys of the HuggingFace config.json that are not model hyper-parameters
    _HF_CONFIG_SERVICE_KEYS = ("architectures", "auto_map", "model_type", "torch_dtype", "transformers_version")

    @classmethod
    def load_from_safetensors(cls, weights_path: str, config_path: Optional[str] = None):
        """Load from a HuggingFace ``MIT-SLS/USAD2-*`` snapshot: ``model.safetensors`` + ``config.json``.

        ``config_path`` defaults to the ``config.json`` next to the weights. The HF wrapper stores the
        encoder under the ``model.`` prefix, which is stripped here.
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.abspath(weights_path)), "config.json")
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        for key in cls._HF_CONFIG_SERVICE_KEYS:
            config.pop(key, None)
        model = cls(make_dataclass("Config", config.keys())(**config))

        state_dict = {
            (k[len("model."):] if k.startswith("model.") else k): v for k, v in _read_safetensors(weights_path).items()
        }
        model.load_state_dict(state_dict, strict=True)
        return model


def load_model(
    model_file_path: Optional[str] = None,
    model_name="usad_small",
    device: Optional[str] = None,
    **kwargs,
) -> UsadModel:
    """Load the USAD model from a checkpoint file.

    Args:
        model_file_path (Optional[str], optional):
            Path to checkpoint. Defaults to None.
        model_name (str, optional):
            Name of model. Defaults to "usad_small".
        device (Optional[str], optional):
            Device to run on. Defaults to None.

    Returns:
        UsadModel: Loaded USAD model instance.
    """
    if device is None:
        torch_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        torch_device = torch.device(device)

    if model_file_path is None or not os.path.exists(model_file_path):
        model_file_path = f"/data/sls/scratch/hengjui/models/{model_name}.pt"

    # Initialize model
    model = UsadModel.load_from_fairseq_ckpt(model_file_path)
    model = model.to(torch_device).eval()

    return model


def get_scene_embeddings(x: torch.Tensor, model: UsadModel) -> torch.Tensor:
    """Get scene embeddings from the model. (instance-level embeddings)

    Args:
        x (torch.Tensor): Input waveform.
        model (UsadModel): USAD model.

    Returns:
        torch.Tensor: Scene embeddings of shape (batch_size, embedding_dim).
    """
    model.eval()
    with torch.no_grad():
        results = model(x)
        embeddings = torch.cat(results["hidden_states"], dim=-1).mean(dim=1)
    return embeddings


def get_timestamp_embeddings(
    x: torch.Tensor, model: UsadModel
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get timestamp embeddings from the model. (frame-level embeddings)

    Args:
        x (torch.Tensor): Input waveform of shape (batch_size, wav_len).
        model (UsadModel): USAD model.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - embeddings: Frame-level embeddings of shape
                (batch_size, seq_len, embedding_dim).
            - timestamps: Timestamps corresponding to each
                frame of shape (batch_size, seq_len).
    """
    model.eval()
    with torch.no_grad():
        results = model(x)
        embeddings = torch.cat(results["hidden_states"], dim=-1)
        timestamps = (
            (torch.arange(embeddings.shape[1]).float() * 20 + 17.5)
            .clamp(0, x.shape[1] / 16)
            .unsqueeze(0)
            .expand(x.shape[0], -1)
        )
    return embeddings, timestamps
