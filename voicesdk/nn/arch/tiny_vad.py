import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing import Optional, Tuple


class TorchSileroVAD(nn.Module):
    def __init__(self):
        super().__init__()

        self.n_fft = 256
        self.stride = 128
        self.pad = 64
        self.cutoff = self.n_fft // 2 + 1

        self.stft_conv = nn.Conv1d(
            in_channels=1,
            out_channels=2 * self.cutoff,  # 258 = 129 real + 129 imag
            kernel_size=self.n_fft,
            stride=self.stride,
            padding=0,
            bias=False,
        )

        self.conv1 = nn.Conv1d(129, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)

        self.lstm_cell = nn.LSTMCell(128, 128)
        self.final_conv = nn.Conv1d(128, 1, kernel_size=1)

    def stft(self, input_data: Tensor) -> Tensor:
        """
        Compute magnitude spectrogram using self.stft_conv.

        Args:
            input_data: Tensor [B, T]

        Returns:
            magnitude: Tensor [B, 129, Frames]
        """
        # Same idea as JIT:
        # input_data0 = unsqueeze(padding.forward(input_data), 1)
        x = F.pad(input_data, (0, self.pad), mode="reflect")
        x = x.unsqueeze(1)  # [B, 1, T]

        # forward_transform = conv1d(...)
        x = self.stft_conv(x)  # [B, 258, Frames]

        # split into real / imag
        real_part = x[:, :self.cutoff, :]
        imag_part = x[:, self.cutoff:, :]

        # magnitude = sqrt(real^2 + imag^2)
        magnitude = torch.sqrt(real_part.pow(2) + imag_part.pow(2))
        return magnitude

    def transform_(self, input_data: Tensor) -> Tensor:
        """
        Alias matching the JIT naming, but returns only magnitude.
        """
        return self.stft(input_data)

    def forward(
        self,
        x: Tensor,
        state: Optional[Tensor | Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: Tensor [B, T]
            state:
                None
                or tuple(h, c), each [B, 128]
                or Tensor [2, B, 128]

        Returns:
            output: Tensor [B, 1]
            state: Tensor [2, B, 128]
        """

        if state is not None:
            if isinstance(state, torch.Tensor):
                h_prev, c_prev = state[:, 0], state[:, 1]
                state = (h_prev, c_prev)

        # Use extracted STFT magnitude
        x = self.stft(x)  # [B, 129, Frames]

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = x.squeeze(-1)  # [B, 128]

        h, c = self.lstm_cell(x, state)

        x = h.unsqueeze(-1)
        state = torch.stack([h, c], dim=0)

        x = F.relu(x)
        x = self.final_conv(x)
        x = torch.sigmoid(x)

        x = x.squeeze(1)
        x = x.mean(dim=1, keepdim=True)

        return x, state.permute((1, 0, 2))


class VADDecoderRNNJIT(nn.Module):

    def __init__(self):
        super(VADDecoderRNNJIT, self).__init__()

        self.rnn = nn.LSTMCell(128, 128)
        self.decoder = nn.Sequential(nn.Dropout(0.1),
                                     nn.ReLU(),
                                     nn.Conv1d(128, 1, kernel_size=1),
                                     nn.Sigmoid())

    def forward(self, x, state=torch.zeros(0)):
        x = x.squeeze(-1)
        if len(state):
            h, c = self.rnn(x, (state[0], state[1]))
        else:
            h, c = self.rnn(x)

        x = h.unsqueeze(-1).float()
        state = torch.stack([h, c])
        x = self.decoder(x)
        return x, state
