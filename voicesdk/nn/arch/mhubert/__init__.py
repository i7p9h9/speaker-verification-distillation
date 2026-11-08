from .mhubert_model import MHubert, MHubertConfig, load_mhubert
from .mhubert_torchaudio import TorchaudioMHubert, load_mhubert_torchaudio

# Linear layers of the HuBERT transformer blocks (substring match, for LoRA targets).
DEFAULT_LORA_TARGETS = "q_proj,k_proj,v_proj,out_proj,intermediate_dense,output_dense"
