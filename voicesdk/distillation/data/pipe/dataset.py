import typing as tp

from ._types import AudioSegments
from .audio_reader import AudioReaderBase


class WavDataset:
    """
    Dataset that maps a list of wav file paths to AudioSegments.
    All audio reading / normalization / segmentation logic is delegated
    to the injected *reader* (an AudioReaderBase subclass).
    """

    def __init__(
        self,
        wav_scp: tp.List[str],
        reader: AudioReaderBase,
    ):
        """
        Args:
            wav_scp: List of paths to wav files
            reader: AudioReaderBase instance that defines how each file is read
        """
        self.wav_list = wav_scp
        self.reader = reader

    def __len__(self) -> int:
        return len(self.wav_list)

    def __getitem__(self, idx: int) -> AudioSegments:
        return self.reader.read(self.wav_list[idx])
