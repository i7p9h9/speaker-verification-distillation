import typing as tp
from pathlib import Path

from torch.utils.data import Dataset

from validation import ValidationTrial
from voicesdk.distillation.data import AudioReaderBase, AudioSegments, WavDataset
from voicesdk.utils import find_files_recursive


class VoxDataset(Dataset):
    """
    Dataset for VoxCeleb-style data with speaker/utterance/file hierarchy.
    Wraps WavDataset internally and adds VoxCeleb-specific indexing.

    Expected directory structure:
        root/
            speaker_id/
                utterance_id/
                    file.wav
    """

    def __init__(
        self,
        root: tp.Union[str, Path],
        reader: AudioReaderBase,
        file_extension: str = "wav",
    ):
        """
        Args:
            root: Root directory containing speaker folders
            reader: AudioReaderBase instance that defines how each file is read
            file_extension: Audio file extension to search for
        """
        super().__init__()

        self.root = Path(root)
        self.reader = reader

        # Find all audio files
        self._wav_list = find_files_recursive(str(self.root), file_extension)

        # Create internal WavDataset
        self._wav_dataset = WavDataset(wav_scp=self._wav_list, reader=reader)

        # Build file_to_index mapping: "spk_name/utt_name/file_name.wav" -> index
        self._file_to_index: tp.Dict[str, int] = {}
        self._speaker_set: tp.Set[str] = set()
        self._utterance_set: tp.Set[str] = set()

        self._build_index(self._wav_list)

    def _build_index(self, wav_list: tp.List[str]) -> None:
        """Build file_to_index mapping and speaker/utterance sets."""
        self._file_to_index.clear()
        self._speaker_set.clear()
        self._utterance_set.clear()

        for idx, filepath in enumerate(wav_list):
            # Extract last 2 folders + filename
            path = Path(filepath)
            relative_key = str(Path(*path.parts[-3:]))
            self._file_to_index[relative_key] = idx

            # Track speakers and utterances
            spk_name = path.parts[-3]
            utt_name = path.parts[-2]
            self._speaker_set.add(spk_name)
            self._utterance_set.add(f"{spk_name}/{utt_name}")

    @property
    def wav_list(self) -> tp.List[str]:
        """Returns list of wav file paths."""
        return self._wav_dataset.wav_list

    @property
    def file_to_index(self) -> tp.Dict[str, int]:
        """Returns mapping from relative file path to dataset index."""
        return self._file_to_index

    @property
    def trial_to_index(self) -> tp.Dict[str, int]:
        return {self._file_to_trial(key): idx for idx, key in enumerate(self._wav_list)}

    @property
    def trials(self) -> tp.List[str]:
        """
        Returns trial identifiers in format: spk_name-utt_name-file_name
        Derived from file_to_index keys.
        """
        result = []
        for key in self._file_to_index.keys():
            result.append(self._file_to_trial(key))
        return result

    @property
    def n_speakers(self) -> int:
        """Returns number of unique speakers."""
        return len(self._speaker_set)

    @property
    def n_utterances(self) -> int:
        """Returns number of unique utterances (speaker/utterance combinations)."""
        return len(self._utterance_set)

    @property
    def n_samples(self) -> int:
        """Returns total number of audio files."""
        return len(self._wav_dataset)

    def __len__(self) -> int:
        return len(self._wav_dataset)

    def __getitem__(self, idx: int) -> AudioSegments:
        """
        Returns audio segments for given index.
        Delegates to internal WavDataset.
        """
        return self._wav_dataset[idx]

    def _file_to_trial(self, file: str) -> str:
        parts = Path(file).with_suffix('').parts
        trial_id = f"{parts[-3]}-{parts[-2]}-{parts[-1]}"
        return trial_id

    def _trial_id_to_file_key(self, trial_id: str) -> str:
        """
        Convert trial ID to file key format.

        Args:
            trial_id: Trial ID in format "spk_name-utt_name-file_name"

        Returns:
            File key in format "spk_name/utt_name/file_name"
        """
        parts = trial_id.split("-", 2)
        if len(parts) != 3:
            raise ValueError(f"Invalid trial_id format: {trial_id}")
        return f"{parts[0]}/{parts[1]}/{parts[2]}.wav"

    def get_by_trial(self, trial_id: str) -> AudioSegments:
        """
        Get audio segments by trial identifier.

        Args:
            trial_id: Trial ID in format "spk_name-utt_name-file_name"

        Returns:
            AudioSegments with processed data
        """
        file_key = self._trial_id_to_file_key(trial_id)

        if file_key not in self._file_to_index:
            raise KeyError(f"Trial not found: {trial_id}")

        idx = self._file_to_index[file_key]
        return self[idx]

    def filter_by_trials(self, trials: tp.List[ValidationTrial]) -> "VoxDataset":
        """
        Create a filtered copy of the dataset containing only samples
        referenced in the provided trials list.

        Args:
            trials: List of ValidationTrial objects containing trial_left and trial_right

        Returns:
            New VoxDataset instance with filtered samples
        """
        # Collect all unique trial IDs from both left and right
        unique_trial_ids: tp.Set[str] = set()
        for trial in trials:
            unique_trial_ids.add(trial.trial_left)
            unique_trial_ids.add(trial.trial_right)

        # Convert trial IDs to file keys and find corresponding file paths
        filtered_wav_list: tp.List[str] = []
        for trial_id in unique_trial_ids:
            try:
                file_key = self._trial_id_to_file_key(trial_id)
                if file_key in self._file_to_index:
                    idx = self._file_to_index[file_key]
                    filtered_wav_list.append(self.wav_list[idx])
            except ValueError:
                # Skip invalid trial IDs
                continue

        # Create a new dataset instance with filtered files
        filtered_dataset = VoxDataset.__new__(VoxDataset)
        filtered_dataset.root = self.root
        filtered_dataset.reader = self.reader
        filtered_dataset._wav_list = filtered_wav_list
        filtered_dataset._wav_dataset = WavDataset(wav_scp=filtered_wav_list, reader=self.reader)

        # Rebuild mappings for filtered dataset
        filtered_dataset._file_to_index = {}
        filtered_dataset._speaker_set = set()
        filtered_dataset._utterance_set = set()
        filtered_dataset._build_index(filtered_wav_list)

        return filtered_dataset
