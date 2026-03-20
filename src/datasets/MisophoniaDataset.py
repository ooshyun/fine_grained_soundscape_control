"""
Torch dataset object for synthetically rendered spatial data.
"""

import os
import logging
import time
import torch
import signal
import threading
import pyloudnorm as pyln
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from src.datasets.augmentations.audio_augmentations import AudioAugmentations
from torch.utils.data import Dataset
from src.datasets.motion_simulator import CIPICMotionSimulator2
from src.datasets.multi_ch_simulator import CIPICSimulator, MultiChSimulator
import numpy as np
import soundfile as sf
import librosa
import traceback as tb
import math
from torchmetrics.functional import signal_noise_ratio as snr

import json
import tqdm
import src.utils as utils
from src.config.paths import get_dataset_root

# Configure logging only once to avoid race conditions
logger = logging.getLogger(__name__)

# Try to import torchaudio for GPU resampling
try:
    import torchaudio

    TORCHAUDIO_AVAILABLE = True
    logger.info("TorchAudio available for GPU resampling")
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    logger.warning("TorchAudio not available, falling back to CPU resampling")

# Try to import scipy for fallback
try:
    import scipy.signal

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("SciPy not available for fallback resampling")

class TensorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        return super(TensorEncoder, self).default(obj)

def get_snr(target, mixture, EPS=1e-9):
    """
    Computes the average SNR across all channels
    """
    return snr(mixture, target).mean()


def scale_noise_to_snr(
    target_speech: torch.Tensor, noise: torch.Tensor, target_snr: float
):
    """
    Rescales a BINAURAL noise signal to achieve an average SNR (across both channels) equal to target snr.
    Let k be the noise scaling factor
    SNR_tgt = (SNR_left_scaled + SNR_right_scaled) / 2 = 0.5 * (10 log(S_L^T S_L/S_N^T S_N) - 20 log(k) + 10 log(S_R^T S_R / N_R^T N_R) - 20 log(k))
            = 0.5 * (SNR_left_unscaled + SNR_right_unscaled - 40 log(k)) = avg_snr_initial - 20 log (k)
    """

    current_snr = get_snr(target_speech, noise + target_speech)

    pwr = (current_snr - target_snr) / 20
    k = 10**pwr

    return k * noise


def pad_audio_to_minimum_length(audio, sample_rate, min_duration=0.4):
    """
    Pad audio to minimum required length for LUFS calculation.
    pyloudnorm requires audio longer than block size (default 400ms).

    Args:
        audio: Audio array
        sample_rate: Sample rate
        min_duration: Minimum duration in seconds (default 0.4 = 400ms)

    Returns:
        padded_audio: Audio padded to minimum length
    """
    min_samples = int(min_duration * sample_rate)
    current_length = len(audio)

    if current_length < min_samples:
        # Pad with zeros at the end
        padding = min_samples - current_length
        padded = np.pad(audio, (0, padding), mode="constant", constant_values=0)
        return padded, True

    return audio, False


def get_lufs(audio, sample_rate):
    # Convert to numpy array if needed (pyloudnorm requires numpy arrays)
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()
    elif not isinstance(audio, np.ndarray):
        audio = np.asarray(audio)

    # Ensure 1D array
    if len(audio.shape) > 1:
        audio = audio.flatten()

    meter = pyln.Meter(sample_rate)
    return meter.integrated_loudness(audio)


def normalize_to_lufs(
    audio,
    sample_rate,
    target_lufs,
    padding=False,
    verbose=False,
    max_iterations=100,
    tolerance=0.1,
):
    """
    Normalize audio to target LUFS level

    Args:
        audio: Audio array (numpy array or torch.Tensor)
        sample_rate: Sample rate
        target_lufs: Target LUFS level
        padding: If True, automatically pad short audio

    Returns:
        (normalized_audio, original_lufs)
    """
    # Convert to numpy array if needed (pyloudnorm requires numpy arrays)
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()
    elif not isinstance(audio, np.ndarray):
        audio = np.asarray(audio)

    # Ensure 1D array
    if len(audio.shape) > 1:
        audio = audio.flatten()

    # Pad if needed for LUFS calculation
    if padding:
        audio, _ = pad_audio_to_minimum_length(audio, sample_rate, min_duration=0.5)

    current_lufs = get_lufs(audio, sample_rate)

    # Normalize (normalization works on any length)
    normalized = pyln.normalize.loudness(audio, current_lufs, target_lufs)

    # Iterative refinement
    for iteration in range(max_iterations):
        # Check convergence
        error = abs(current_lufs - target_lufs)
        if error < tolerance:
            if verbose and iteration > 0:
                logger.debug(
                    f"Loudness normalization is converged after {iteration + 1} iterations"
                )
            return normalized, current_lufs

        # Update current LUFS
        current_lufs = get_lufs(normalized, sample_rate)
        if verbose:
            logger.debug(
                f"Iteration {iteration}: Actual LUFS = {current_lufs:.4f}, "
                f"Target = {target_lufs:.4f}, Error = {error:.4f} dB"
            )
        # Apply correction
        normalized = pyln.normalize.loudness(normalized, current_lufs, target_lufs)

    return normalized, current_lufs


import re


def glob_re(pattern, strings):
    return filter(re.compile(pattern).match, strings)


class MisophoniaDataset(Dataset):
    def __init__(
        self,
        fg_sounds_dir,
        bg_sounds_dir,
        hrtf_list,
        split,
        noise_sounds_dir="",
        duration=5,
        sr=44100,
        hrtf_type="CIPIC",
        motion_use_piecewise_arcs=False,
        augmentations=[],
        samples_per_epoch=20000,
        use_motion=False,
        num_total_labels=20,
        num_fg_sounds_min=1,
        num_fg_sounds_max=5,
        num_bg_sounds_min=1,
        num_bg_sounds_max=3,
        num_noise_sounds_min=1,
        num_noise_sounds_max=1,
        num_output_channels=2,
        snr_range_fg=[5, 15],
        snr_range_bg=[0, 10],
        ref_db=-50,
        bg_sounds=None,
        onflight_mode=0,
        use_torchaudio_resampling=True,
        resampling_timeout=10.0,
        target_multichannel=False,
        root_dataset_dir=None,
    ) -> None:
        super().__init__()
        assert split in [
            "train",
            "val",
            "test",
        ], "`split` must be one of ['train', 'val', 'test']"

        # Set root_dataset_dir with priority: parameter > env variable > default
        if root_dataset_dir is None:
            root_dataset_dir = get_dataset_root()

        # Use thread-local storage for power threshold to avoid race conditions
        self.pwr_threshold = -40
        self.default_pwr_threshold = self.pwr_threshold
        self.lufs_verification_threshold = 0.1
        self.fg_sounds_dir = (
            fg_sounds_dir
            if root_dataset_dir in fg_sounds_dir
            else os.path.join(root_dataset_dir, fg_sounds_dir)
        )
        self.bg_sounds_dir = (
            bg_sounds_dir
            if root_dataset_dir in bg_sounds_dir
            else os.path.join(root_dataset_dir, bg_sounds_dir)
        )
        self.noise_sounds_dir = (
            noise_sounds_dir
            if root_dataset_dir in noise_sounds_dir
            else os.path.join(root_dataset_dir, noise_sounds_dir)
        )
        if isinstance(hrtf_list, str):
            self.hrtf_list = (
                hrtf_list
                if root_dataset_dir in hrtf_list
                else os.path.join(root_dataset_dir, hrtf_list)
            )
        elif isinstance(hrtf_list, list):
            self.hrtf_list = (
                hrtf_list
                if root_dataset_dir in hrtf_list[0]
                else [os.path.join(root_dataset_dir, _hrtf_list) for _hrtf_list in hrtf_list]
            )
        self.split = split
        self.sr = sr
        self.onflight_mode = onflight_mode

        # GPU resampling configuration
        self.use_torchaudio_resampling = (
            use_torchaudio_resampling
            and TORCHAUDIO_AVAILABLE
            and torch.cuda.is_available()
        )
        self.resampling_timeout = resampling_timeout

        # Initialize GPU resampling if available
        if self.use_torchaudio_resampling:
            try:
                # Set device for resampling (use CPU for multiprocessing safety)
                self.resample_device = torch.device("cpu")  # Safer for multiprocessing
                logger.debug(
                    f"GPU resampling enabled with device: {self.resample_device}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to initialize GPU resampling, falling back to CPU: {e}"
                )
                self.use_torchaudio_resampling = False
        else:
            logger.debug("Using CPU resampling")

        # Configure multiprocessing settings for better performance
        self._configure_multiprocessing()

        self.duration = duration
        self.mix_snr_range = [-10, 10]
        self.num_fg_sounds_range = [num_fg_sounds_min, num_fg_sounds_max]
        self.num_bg_sounds_range = [num_bg_sounds_min, num_bg_sounds_max]
        self.num_noise_sounds_range = [num_noise_sounds_min, num_noise_sounds_max]
        self.snr_range_fg = snr_range_fg
        self.snr_range_bg = snr_range_bg
        self.num_output_channels = num_output_channels
        self.target_multichannel = target_multichannel # output one channel process every speaker

        self.samples_per_epoch = samples_per_epoch
        self.ref_db = ref_db

        # Data augmentation
        self.perturbations = AudioAugmentations(augmentations)

        logging.info(f"  - FG directory: {fg_sounds_dir}")
        logging.info(f"  - BG directory: {bg_sounds_dir}")
        logging.info(f"  - Noise directory: {noise_sounds_dir}")
        logging.info(f"  - HRTF directory: {hrtf_list}")

        # HRTF simulator with motion
        if hrtf_type == "CIPIC":
            self.multi_ch_simulator = CIPICSimulator(self.hrtf_list, sr)
        elif hrtf_type == "MultiCh":
            if use_motion:
                cipic_simulator_type = lambda sofa, sr: CIPICMotionSimulator2(
                    sofa, sr, use_piecewise_arcs=motion_use_piecewise_arcs
                )
            else:
                cipic_simulator_type = CIPICSimulator
            self.multi_ch_simulator = MultiChSimulator(
                self.hrtf_list, sr, cipic_simulator_type, dset=split
            )
            # self.multi_ch_simulator = MultiChSimulatorSemHL(
            #     self.hrtf_list, sr, cipic_simulator_type, dset=split
            # )
            # import src.datasets.multi_ch_simulator as multi_ch_simulator_semhl
            # self.multi_ch_simulator = multi_ch_simulator_semhl.Multi_Ch_Simulator(
            #     self.hrtf_list, sr, cipic_simulator_type, dset=split
            # )
        else:
            raise NotImplementedError

        # FG/BG Sounds
        self.fg_sounds = sorted(os.listdir(self.fg_sounds_dir))

        # shoh: test 1spk with embedding layer (251016)
        # self.fg_sounds = sorted(["baby_cry",
        #                         "cat",
        #                         "cock_a_doodle_doo",
        #                         "cricket",
        #                         "dog"])
        if len(self.fg_sounds) > num_total_labels:
            self.fg_sounds = self.fg_sounds[:num_total_labels]
        elif len(self.fg_sounds) == num_total_labels:
            pass
        else:
            raise ValueError(
                f"Not enough FG sounds found in {self.fg_sounds_dir}. \
                Expected at least {num_total_labels} but found {len(self.fg_sounds)}. \
                Please increase the number of FG sounds or decrease the number of total labels."
            )

        self.bg_sounds = sorted(os.listdir(self.bg_sounds_dir))
        self.noise_sounds = sorted(os.listdir(self.noise_sounds_dir))

        if bg_sounds is not None:
            self.bg_sounds = [x for x in self.bg_sounds if x in bg_sounds]

        logging.info(
            f"Foreground sounds: {self.fg_sounds}, number of sounds: {len(self.fg_sounds)}"
        )
        logging.info(
            f"Background sounds: {self.bg_sounds}, number of sounds: {len(self.bg_sounds)}"
        )
        logging.info(
            f"Noise sounds: {self.noise_sounds}, number of sounds: {len(self.noise_sounds)}"
        )

    def _configure_multiprocessing(self):
        """
        Configure multiprocessing settings to prevent stuck processes and improve performance.
        """
        try:
            # Limit PyTorch threads to prevent resource contention
            torch.set_num_threads(1)

            # Set environment variables for better multiprocessing
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            os.environ["NUMEXPR_NUM_THREADS"] = "1"

            logger.debug("Multiprocessing configuration applied: limited threads to 1")

        except Exception as e:
            logger.warning(f"Failed to configure multiprocessing settings: {e}")

    def __len__(self):
        return self.samples_per_epoch

    def read_audio_as_mono(self, sf: sf.SoundFile, num_frames: int):
        # Workaround for loading flac files
        if sf.name.endswith(".flac"):
            audio = sf.read(frames=num_frames, dtype="int32")
            audio = (audio / (2 ** (31) - 1)).astype(np.float32)
        else:
            audio = sf.read(frames=num_frames, dtype="float32")

        # If multichannel audio take a single channel
        if len(audio.shape) > 1:
            audio = audio[:, 0]

        return audio

    # ensure the appended audio is not silent.
    def check_silence(self, audio, threshold=1e-3):
        return np.sqrt(np.mean(audio**2)) < threshold

    def compute_power_timestamps(
        self, audio, sr=44100, window_size=128, hop_size=64, threshold_db=-30, eps=1e-9
    ):
        """
        Computes power (dB) of an audio signal

        Parameters:
        - audio: Input audio waveform(np.array)
        - sr (int): Sample rate of the audio
        - threshold_db (float): Silence threshold in dB (default: -30 dB).
        - eps (float): Small value to prevent log(0) issues.

        Returns:
        - timestamps (list of float): Time (in seconds) where power > threshold.
        - power_db (np.array): Power in dB over time.
        """
        num_windows = (len(audio) - window_size) // hop_size + 1
        power_db = np.zeros(num_windows)

        for i in range(num_windows):
            start = i * hop_size
            end = start + window_size
            window = audio[start:end]

            # Compute power: 10 * log10(mean(signal^2) + eps)
            power_db[i] = 10 * np.log10(np.mean(window**2) + eps)

        # Generate timestamps
        timestamps = np.arange(num_windows) * (hop_size / sr)

        # Filter timestamps where power > threshold
        valid_timestamps = timestamps[power_db > threshold_db]

        return valid_timestamps.tolist(), power_db

    def read_audio(self, sf: sf.SoundFile, num_frames: int, as_mono=True):
        # Workaround for loading flac files
        if sf.name.endswith(".flac"):
            audio = sf.read(frames=num_frames, dtype="int32")
            audio = (audio / (2 ** (31) - 1)).astype(np.float32)
        else:
            audio = sf.read(frames=num_frames, dtype="float32")

        # If multichannel audio take a single channel
        if (len(audio.shape) > 1) and (as_mono):
            audio = audio[:, 0]

        return audio

    def _gpu_resample_with_timeout(self, audio, orig_sr, target_sr, timeout=None):
        """
        GPU-accelerated resampling with timeout protection for multiprocessing safety.
        """
        if timeout is None:
            timeout = self.resampling_timeout

        def _resample_worker():
            try:
                # Convert to torch tensor
                if isinstance(audio, np.ndarray):
                    audio_tensor = torch.from_numpy(audio).float()
                else:
                    audio_tensor = audio.float()

                # Ensure correct shape for torchaudio
                if len(audio_tensor.shape) == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)  # Add channel dimension

                # Move to device
                audio_tensor = audio_tensor.to(self.resample_device)

                # Use torchaudio resampling
                resampler = torchaudio.transforms.Resample(  # check this with librosa orignal resample result
                    orig_freq=orig_sr,
                    new_freq=target_sr,
                    resampling_method="sinc_interp_hann",
                ).to(
                    self.resample_device
                )

                resampled = resampler(audio_tensor)
                logger.debug(
                    f"Resampled audio shape: {resampled.shape}, type: {type(resampled)}"
                )
                # Convert back to numpy
                if resampled.is_cuda:
                    resampled = resampled.cpu()

                return resampled.squeeze(0).numpy()  # Remove channel dimension

            except Exception as e:
                logger.warning(f"GPU resampling failed: {e}")
                raise e

        try:
            # Use ThreadPoolExecutor with timeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_resample_worker)
                result = future.result(timeout=timeout)
                return result
        except FutureTimeoutError:
            logger.warning(f"GPU resampling timed out after {timeout}s")
            raise TimeoutError(f"Resampling timed out after {timeout}s")
        except Exception as e:
            logger.warning(f"GPU resampling error: {e}")
            raise e

    def _cpu_resample_with_timeout(self, audio, orig_sr, target_sr, timeout=None):
        """
        CPU resampling with timeout protection using librosa or scipy.
        """
        if timeout is None:
            timeout = self.resampling_timeout

        def _resample_worker():
            try:
                # Try librosa first
                return librosa.resample(
                    audio, orig_sr=orig_sr, target_sr=target_sr, res_type="kaiser_best"
                )
            except Exception as librosa_error:
                logger.debug(f"Librosa resampling failed: {librosa_error}")
                if SCIPY_AVAILABLE:
                    try:
                        # Fallback to scipy
                        return scipy.signal.resample(
                            audio, int(len(audio) * target_sr / orig_sr)
                        )
                    except Exception as scipy_error:
                        logger.error(
                            f"Both librosa and scipy resampling failed: {scipy_error}"
                        )
                        raise scipy_error
                else:
                    raise librosa_error

        try:
            # Use ThreadPoolExecutor with timeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_resample_worker)
                result = future.result(timeout=timeout)
                return result
        except FutureTimeoutError:
            logger.warning(f"CPU resampling timed out after {timeout}s")
            raise TimeoutError(f"Resampling timed out after {timeout}s")
        except Exception as e:
            logger.warning(f"CPU resampling error: {e}")
            raise e

    def sample_snippet(
        self, audio_file, rng: np.random.RandomState, as_mono=True
    ) -> np.ndarray:
        # Start overall timing
        start_time = time.time()
        logger.debug(
            f"MisophoniaDataset(sample_snippet): Starting sampling snippet from {audio_file}"
        )

        try:
            # Phase 1: File existence check
            phase1_start = time.time()
            if not os.path.exists(audio_file):
                logger.warning(
                    f"MisophoniaDataset(sample_snippet): File does not exist: {audio_file}"
                )
                return None
            phase1_time = time.time() - phase1_start
            logger.debug(
                f"MisophoniaDat`aset(sample_snippet): Phase 1 (file check) took {phase1_time:.4f}s"
            )

            # Phase 2: File opening and metadata reading
            phase2_start = time.time()
            try:
                with sf.SoundFile(audio_file) as f:
                    file_sr = f.samplerate
                    num_frames = f.frames
                    phase2_time = time.time() - phase2_start
                    logger.debug(
                        f"MisophoniaDataset(sample_snippet): Phase 2 (file open + metadata) took {phase2_time:.4f}s"
                    )
                    logger.debug(
                        f"MisophoniaDataset(sample_snippet): File properties: {file_sr}Hz, {num_frames} frames"
                    )

                    # Validate file properties
                    if file_sr <= 0 or num_frames <= 0:
                        logger.warning(
                            f"MisophoniaDataset(sample_snippet): Invalid file properties: {audio_file}"
                        )
                        return None

                    # Phase 3: Frame calculation and audio reading strategy
                    phase3_start = time.time()
                    total_frames = math.ceil(self.duration * file_sr)
                    logger.debug(
                        f"MisophoniaDataset(sample_snippet): Total frames needed: {total_frames}"
                    )

                    if total_frames > num_frames:
                        # Read entire audio
                        read_start = time.time()
                        audio = self.read_audio(
                            f, num_frames=num_frames, as_mono=as_mono
                        ).T
                        read_time = time.time() - read_start
                        logger.debug(
                            f"MisophoniaDataset(sample_snippet): Audio read took {read_time:.4f}s"
                        )

                        if len(audio.shape) > 1:
                            assert (
                                audio.shape[0] == 2
                            ), "If multichannel, must be binaural"

                        # Pad zeros to get to target duration
                        pad_start = time.time()
                        remain = total_frames - num_frames
                        pad_front = rng.randint(0, remain)

                        if len(audio.shape) > 1:
                            audio = np.pad(
                                audio, ((0, 0), (pad_front, remain - pad_front))
                            )
                        else:
                            audio = np.pad(audio, (pad_front, remain - pad_front))
                        pad_time = time.time() - pad_start
                        logger.debug(
                            f"MisophoniaDataset(sample_snippet): Padding took {pad_time:.4f}s"
                        )

                    else:
                        # Randomly choose start of snippet
                        seek_start = time.time()
                        start_frame = rng.randint(0, num_frames - total_frames + 1)
                        f.seek(start_frame)
                        seek_time = time.time() - seek_start
                        logger.debug(
                            f"MisophoniaDataset(sample_snippet): Seek to frame {start_frame} took {seek_time:.4f}s"
                        )

                        # Move to start of snippet and read
                        read_start = time.time()
                        audio = self.read_audio(
                            f, num_frames=total_frames, as_mono=as_mono
                        ).T
                        read_time = time.time() - read_start
                        logger.debug(
                            f"MisophoniaDataset(sample_snippet): Audio read took {read_time:.4f}s"
                        )

                    phase3_time = time.time() - phase3_start
                    logger.debug(
                        f"MisophoniaDataset(sample_snippet): Phase 3 (frame calc + audio read) took {phase3_time:.4f}s"
                    )
                    logger.debug(
                        f"MisophoniaDataset(sample_snippet): Audio shape after read: {audio.shape}"
                    )

                    if len(audio.shape) > 1:
                        assert audio.shape[0] == 2, "If multichannel, must be binaural"

                    # Phase 4: Audio validation
                    phase4_start = time.time()
                    if audio is None or audio.size == 0:
                        logger.warning(
                            f"MisophoniaDataset(sample_snippet): Empty audio data from: {audio_file}"
                        )
                        return None

                    assert (
                        audio.shape[-1] == total_frames
                    ), f"Number of samples in audio incorrect. Expected {audio.shape[-1]} found {total_frames}."
                    phase4_time = time.time() - phase4_start
                    logger.debug(
                        f"MisophoniaDataset(sample_snippet): Phase 4 (validation) took {phase4_time:.4f}s"
                    )

                    # Phase 5: Optimized Resampling with timeout protection
                    phase5_start = time.time()
                    try:
                        resample_start = time.time()

                        # Choose resampling method based on configuration
                        if self.use_torchaudio_resampling:
                            logger.debug(
                                f"MisophoniaDataset(sample_snippet): Using GPU resampling for {audio_file}"
                            )
                            audio = self._gpu_resample_with_timeout(
                                audio, file_sr, self.sr
                            )
                            resample_method = "GPU"
                        else:
                            logger.debug(
                                f"MisophoniaDataset(sample_snippet): Using CPU resampling for {audio_file}"
                            )
                            audio = self._cpu_resample_with_timeout(
                                audio, file_sr, self.sr
                            )
                            resample_method = "CPU"

                        resample_time = time.time() - resample_start
                        logger.debug(
                            f"MisophoniaDataset(sample_snippet): {resample_method} resampling took {resample_time:.4f}s"
                        )

                    except TimeoutError as timeout_error:
                        logger.warning(
                            f"MisophoniaDataset(sample_snippet): Resampling timed out for {audio_file}: {timeout_error}"
                        )
                        return None
                    except Exception as resample_error:
                        logger.warning(
                            f"MisophoniaDataset(sample_snippet): Resampling failed for {audio_file}: {resample_error}"
                        )
                        # Try fallback method
                        try:
                            fallback_start = time.time()
                            if self.use_torchaudio_resampling:
                                # Try CPU fallback
                                logger.debug(
                                    f"MisophoniaDataset(sample_snippet): Trying CPU fallback for {audio_file}"
                                )
                                audio = self._cpu_resample_with_timeout(
                                    audio, file_sr, self.sr, timeout=5.0
                                )
                                fallback_method = "CPU_fallback"
                            else:
                                # Try simple scipy fallback
                                logger.debug(
                                    f"MisophoniaDataset(sample_snippet): Trying scipy fallback for {audio_file}"
                                )
                                if SCIPY_AVAILABLE:
                                    audio = scipy.signal.resample(
                                        audio, int(len(audio) * self.sr / file_sr)
                                    )
                                    fallback_method = "scipy_fallback"
                                else:
                                    raise Exception(
                                        "No fallback resampling method available"
                                    )

                            fallback_time = time.time() - fallback_start
                            logger.debug(
                                f"MisophoniaDataset(sample_snippet): {fallback_method} took {fallback_time:.4f}s"
                            )

                        except Exception as fallback_error:
                            logger.error(
                                f"MisophoniaDataset(sample_snippet): All resampling methods failed for {audio_file}: {fallback_error}"
                            )
                            return None

                    phase5_time = time.time() - phase5_start
                    logger.debug(
                        f"MisophoniaDataset(sample_snippet): Phase 5 (resampling) took {phase5_time:.4f}s"
                    )

                    # Phase 6: Final trimming and validation
                    phase6_start = time.time()
                    tgt_samples = int(self.sr * self.duration)
                    audio = audio[..., :tgt_samples]

                    # Final validation
                    if audio is None or audio.size == 0:
                        logger.warning(
                            f"MisophoniaDataset(sample_snippet): Final audio is empty from: {audio_file}"
                        )
                        return None
                    phase6_time = time.time() - phase6_start
                    logger.debug(
                        f"MisophoniaDataset(sample_snippet): Phase 6 (trimming + final validation) took {phase6_time:.4f}s"
                    )

                # Overall timing summary
                total_time = time.time() - start_time
                logger.debug(
                    f"MisophoniaDataset(sample_snippet): COMPLETED {audio_file} in {total_time:.4f}s "
                    f"(Phase breakdown: file_check={phase1_time:.4f}s, "
                    f"file_open={phase2_time:.4f}s, "
                    f"audio_read={phase3_time:.4f}s, "
                    f"validation={phase4_time:.4f}s, "
                    f"resampling={phase5_time:.4f}s, "
                    f"trimming={phase6_time:.4f}s)"
                )

                return audio

            except Exception as file_error:
                file_error_time = time.time() - start_time
                logger.warning(
                    f"MisophoniaDataset(sample_snippet): File access error for {audio_file} after {file_error_time:.4f}s: {file_error}"
                )
                return None

        except Exception as e:
            error_time = time.time() - start_time
            logger.error(
                f"MisophoniaDataset(sample_snippet): Unexpected error in sample_snippet for {audio_file} after {error_time:.4f}s: {str(e)}"
            )
            return None

    def get_random_src_snippet(
        self, src_path, rng: np.random.RandomState, threshold=True
    ):
        """
        Get a random audio snippet from a source directory.
        It tries to sample non-slient audio snippet depending on the threshold.
        But, some samples have most samples having power below the threshold (pwr_threshold).
        In such cases, it has to reduce the threshold and try again.
        """
        src_audio_list = [
            os.path.join(src_path, x)
            for x in glob_re(".*\.[wav|flac]", os.listdir(src_path))
        ]

        # Check if we have any audio files
        if not src_audio_list:
            logger.warning(f"MisophoniaDataset: No audio files found in {src_path}")
            return None

        # logger.debug(f"MisophoniaDataset: Source audio list: {src_audio_list}")
        n_attempts = 100  # Reduced from 1000 to prevent infinite loops
        cnt_attempts = 0
        pwr_threshold = self.pwr_threshold
        while True:
            src_audio_path = rng.choice(src_audio_list, 1)[0]
            logger.debug(
                f"MisophoniaDataset: Choosing random src audio: {src_audio_path}"
            )
            _attempts = 10
            while _attempts > 0:
                audio = self.sample_snippet(src_audio_path, rng)
                logger.debug(
                    f"MisophoniaDataset: Audio: {src_audio_path} - {audio.shape}"
                )

                # Handle case where sample_snippet returns None due to file reading errors
                if audio is None:
                    logger.debug(
                        f"MisophoniaDataset: Failed to read audio file: {src_audio_path}"
                    )
                    break  # Try a different file

                if not threshold:
                    return audio

                pwr_dB = 10 * np.log10(np.mean(audio**2) + 1e-9)
                logger.debug(
                    f"MisophoniaDataset: Power dB of src audio: {pwr_dB} <-> threshold: {pwr_threshold}"
                )
                if pwr_dB > pwr_threshold:
                    logger.debug(f"MisophoniaDataset: Returning src audio")
                    return audio
                _attempts -= 1

            cnt_attempts += 1
            if cnt_attempts > n_attempts:
                logging.warning(
                    f"MisophoniaDataset: Failed to sample a valid source {src_audio_path} audio after {n_attempts} attempts"
                )
                # Use thread-local threshold adjustment to avoid race conditions
                pwr_threshold = pwr_threshold - 1
                cnt_attempts = 0

    def choose_random_fg_audio(self, src_label, rng: np.random.RandomState):
        src_path = os.path.join(self.fg_sounds_dir, src_label)
        return self.get_random_src_snippet(src_path, rng)

    def choose_random_bg_audio(self, src_label, rng: np.random.RandomState):
        src_path = os.path.join(self.bg_sounds_dir, src_label)
        return self.get_random_src_snippet(src_path, rng)

    def choose_random_noise_audio(self, src_label, rng: np.random.RandomState):
        src_path = os.path.join(self.noise_sounds_dir, src_label)
        return self.get_random_src_snippet(src_path, rng, threshold=False)

    def get_random_fg_audio(self, rng: np.random.RandomState, n_fg_sources: int):
        fg_audios = []
        fg_onehot = torch.zeros(len(self.fg_sounds))
        valid_fg_labels = []  # Keep only valid FG labels
        valid_src_ids = []  # Store source IDs of actual sounds used
        fg_labels = []
        while len(fg_labels) < n_fg_sources:
            src_id = rng.randint(0, len(self.fg_sounds))
            if src_id in valid_src_ids:
                continue
            tmp_fg_label = self.fg_sounds[src_id]
            audio = self.choose_random_fg_audio(tmp_fg_label, rng)
            if audio is not None:
                fg_audios.append(audio)
                fg_onehot[src_id] = 1  # Update FG one-hot vector
                valid_fg_labels.append(tmp_fg_label)
                valid_src_ids.append(src_id)
                fg_labels.append(tmp_fg_label)
            else:
                logger.warning(
                    f"MisophoniaDataset: Failed to load FG audio for {tmp_fg_label}, skipping"
                )
                continue
        return fg_audios, fg_labels, fg_onehot, valid_fg_labels, valid_src_ids

    def get_random_bg_audio(
        self,
        sound_labels: list[str],
        rng: np.random.RandomState,
        n_bg_sources: int,
        func_choose_audio: callable,
    ):
        """
        Get interfering BG sounds
        """
        bg_labels = []
        bg_audios = []

        while len(bg_labels) < n_bg_sources:
            src_label = rng.choice(sound_labels, size=1)[0]
            logger.debug(f"MisophoniaDataset: Choosing random BG audio: {src_label}")
            audio = func_choose_audio(src_label, rng)

            if audio is not None:
                bg_labels.append(src_label)
                bg_audios.append(audio)
            else:
                logger.warning(
                    f"MisophoniaDataset: Failed to load BG audio for {src_label}, skipping"
                )
                continue
        return bg_audios, bg_labels

    def create_scene(self, rng: np.random.RandomState):
        # Sample number of FG sources
        n_fg_sources = rng.randint(
            self.num_fg_sounds_range[0], self.num_fg_sounds_range[1] + 1
        )
        n_bg_sources = rng.randint(
            self.num_bg_sounds_range[0], self.num_bg_sounds_range[1] + 1
        )
        n_noise_sources = rng.randint(
            self.num_noise_sounds_range[0], self.num_noise_sounds_range[1] + 1
        )
        logger.debug(
            f"MisophoniaDataset: Sampling number of FG sources: {n_fg_sources}"
        )
        logger.debug(
            f"MisophoniaDataset: Sampling number of BG sources: {n_bg_sources}"
        )
        logger.debug(
            f"MisophoniaDataset: Sampling number of noise sources: {n_noise_sources}"
        )

        # Sample and load FG source audio, ensuring it's not silent
        fg_audio, fg_labels, fg_onehot, valid_fg_labels, valid_src_ids = (
            self.get_random_fg_audio(rng, n_fg_sources)
        )

        logger.debug(f"MisophoniaDataset: Choosing random FG audio")
        logger.debug(f"MisophoniaDataset: Valid FG labels: {valid_fg_labels}")
        logger.debug(f"MisophoniaDataset: Valid src ids: {valid_src_ids}")

        # Sample and load BG source audio
        logger.debug(
            f"MisophoniaDataset: Sampling and loading BG source audio: {n_bg_sources}"
        )

        # Get interfering sounds = BG sounds
        bg_audio, bg_labels = self.get_random_bg_audio(
            sound_labels=self.bg_sounds,
            rng=rng,
            n_bg_sources=n_bg_sources,
            func_choose_audio=self.choose_random_bg_audio,
        )
        logger.debug(f"MisophoniaDataset: BG labels: {bg_labels}")
        logger.debug(f"MisophoniaDataset: BG audio: {bg_audio}")

        # Load random noise file
        noise_audio, noise_labels = self.get_random_bg_audio(
            sound_labels=self.noise_sounds,
            rng=rng,
            n_bg_sources=n_noise_sources,
            func_choose_audio=self.choose_random_noise_audio,
        )
        noise_audio = noise_audio[0]  # TODO(shoh): handle multiple noise sources
        noise_labels = noise_labels[0]  # TODO(shoh): bug, fix
        logger.debug(f"MisophoniaDataset: Noise labels: {noise_labels}")
        logger.debug(f"MisophoniaDataset: Noise audio: {noise_audio}")

        # Apply SNR to FG and BG audio relatively to the Noise audio
        # TODO(shoh): add SNR in metadata
        fg_snr = np.random.uniform(self.snr_range_fg[0], self.snr_range_fg[1])
        bg_snr = np.random.uniform(self.snr_range_bg[0], self.snr_range_bg[1])
        fg_lufs = self.ref_db + fg_snr
        bg_lufs = self.ref_db + bg_snr

        for idx, audio in enumerate(fg_audio):
            fg_audio[idx], _ = normalize_to_lufs(
                audio,
                sample_rate=self.sr,
                target_lufs=fg_lufs,
                padding=False,
                max_iterations=100,
                tolerance=self.lufs_verification_threshold,
            )
        for idx, audio in enumerate(bg_audio):
            bg_audio[idx], _ = normalize_to_lufs(
                audio,
                sample_rate=self.sr,
                target_lufs=bg_lufs,
                padding=False,
                max_iterations=100,
                tolerance=self.lufs_verification_threshold,
            )
        noise_audio, _ = normalize_to_lufs(
            noise_audio,
            sample_rate=self.sr,
            target_lufs=self.ref_db,
            padding=False,
            max_iterations=100,
            tolerance=self.lufs_verification_threshold,
        )

        logger.debug(f"MisophoniaDataset: FG audio SNR: {fg_snr}, LUFS: {fg_lufs}")
        logger.debug(f"MisophoniaDataset: BG audio SNR: {bg_snr}, LUFS: {bg_lufs}")
        logger.debug(f"MisophoniaDataset: Noise audio LUFS: {self.ref_db}")

        if logger.level == logging.DEBUG:
            # Verification for LUFS
            for idx, audio in enumerate(fg_audio):
                assert (
                    np.abs(get_lufs(audio, self.sr) - fg_lufs)
                    < self.lufs_verification_threshold
                ), (
                    f"FG audio LUFS verification failed for {audio} at index {idx}, label {valid_fg_labels[idx]}. \n"
                    f"Expected LUFS: {fg_lufs}, Actual LUFS: {get_lufs(audio, self.sr)}"
                )
            for idx, audio in enumerate(bg_audio):
                assert (
                    np.abs(get_lufs(audio, self.sr) - bg_lufs)
                    < self.lufs_verification_threshold
                ), (
                    f"BG audio LUFS verification failed for {audio} at index {idx}, label {bg_labels[idx]}. \n"
                    f"Expected LUFS: {bg_lufs}, Actual LUFS: {get_lufs(audio, self.sr)}"
                )
            assert (
                np.abs(get_lufs(noise_audio, self.sr) - self.ref_db)
                < self.lufs_verification_threshold
            ), (
                f"Noise audio LUFS verification failed for {noise_audio}, label {noise_labels}. \n"
                f"Expected LUFS: {self.ref_db}, Actual LUFS: {get_lufs(noise_audio, self.sr)}"
            )
            logger.debug(f"MisophoniaDataset: LUFS verification passed")

        # Simulate spatialized sources
        seed = rng.randint(1, 1000000)
        bi_srcs, bi_noise = self.multi_ch_simulator.simulate(
            fg_audio + bg_audio, noise_audio, seed
        )

        # Create mixture and ground truth (GT)
        mixture = sum(bi_srcs) + bi_noise
        mixture = torch.from_numpy(mixture).float()
        if self.target_multichannel:
            gt = torch.zeros((n_fg_sources, mixture.shape[-1]))
        else:
            gt = torch.zeros((self.num_output_channels, mixture.shape[-1]))
        
        logger.debug(f"Creating mixture and ground truth (GT)")

        # Assign FG sources to GT, ensuring correct shape
        idx_valid_src_ids_rank = {}
        tmp_valid_src_ids_sorted = sorted(valid_src_ids)
        for idx, src_id in enumerate(tmp_valid_src_ids_sorted):
            idx_valid_src_ids_rank[src_id] = idx

        for fg_sound, src_id in zip(bi_srcs[: len(valid_fg_labels)], valid_src_ids):
            audio = np.mean(fg_sound, axis=0)
            if np.abs(self.num_output_channels - len(fg_onehot)) < 1e-2:
                gt[src_id] = torch.from_numpy(audio).float()
            else:
                gt[idx_valid_src_ids_rank[src_id]] = torch.from_numpy(audio).float()

        logger.debug(f"Assigning FG sources to GT, ensuring correct shape")

        # Convert BG labels to strings for compatibility
        bg_labels = [str(label) for label in bg_labels]
        logger.debug(f"BG labels: {bg_labels}")

        # Convert noise labels to strings for compatibility
        # noise_labels = [str(label) for label in noise_labels] # TODO(shoh): multiple noise sources
        logger.debug(f"Noise labels: {noise_labels}")

        assert (
            mixture.shape[-1] == gt.shape[-1]
        ), f"Mixture and GT have different lengths: {mixture.shape[-1]} != {gt.shape[-1]}"

        return mixture, gt, fg_onehot, valid_fg_labels, bg_labels, noise_labels

    def __getitem__(self, idx):
        start = time.time()
        try:
            logger.debug(
                f"MisophoniaDataset(__getitem__): Creating scene for sample {idx}"
            )
            if self.split == "train":
                # IT IS ACTUALLY **** EXTREMELY **** IMPORTANT TO ADD IDX, ESPECIALLY IF WE ARE FIXING THE WORKERS SEEDS
                # OTHERWISE ALL WORKERS WILL HAVE THE SAME SEED!!!
                seed = idx + np.random.randint(1000000)
                # seed = idx + (idx % 1000000) + 42
            else:
                seed = idx

            logger.debug(
                f"MisophoniaDataset(__getitem__): Setting random seed to {seed}"
            )
            rng = np.random.RandomState(seed)
            # print(f"[DEBUG] Fetching sample {idx}...")
            # Create mixture

            logger.debug(f"MisophoniaDataset(__getitem__): Creating mixture and target")
            mixture, target, one_hot, fg_labels, bg_labels, noise_labels = (
                self.create_scene(rng)
            )
            n_fg = len(fg_labels)
            n_bg = len(bg_labels)
            n_noise = len(noise_labels)

            logger.debug(
                f"MisophoniaDataset(__getitem__): Sample {idx} retrieved successfully: mixture {mixture.shape}, target {target.shape}"
            )

            # Sanity checks
            assert torch.sum(one_hot) == len(
                set(fg_labels)
            ), f"One-hot sum: {torch.sum(one_hot)} != len(set(fg_labels)): {len(set(fg_labels))}"
            # print(f"[DEBUG] fg_labels before padding in __getitem__: {fg_labels}")

            logger.debug(
                f"MisophoniaDataset(__getitem__): Padding fg_labels to fixed max_fg slots"
            )
            fg_labels = fg_labels + [
                "None" for i in range(self.num_fg_sounds_range[1] - n_fg)
            ]
            bg_labels = bg_labels + [
                "None" for i in range(self.num_bg_sounds_range[1] - n_bg)
            ]
            # TODO(shoh): multiple noise sources
            # noise_labels = noise_labels + [
            #     "None" for i in range(self.num_noise_sounds_range[1] - n_noise)
            # ]

            # print(f"[DEBUG] fg_labels after padding in __getitem__: {fg_labels}")
            # print(bg_labels)

            # Apply perturbations to entire audio
            if self.split == "train":
                logger.debug(
                    f"MisophoniaDataset(__getitem__): Applying perturbations to entire audio"
                )
                mixture, target = self.perturbations.apply_random_augmentations(
                    mixture, target, rng
                )

            # print(f"Mixture shape: {mixture.shape}")  # Debug mixture tensor
            # print(f"Target shape: {target.shape}")    # Debug target tensor
            # print(f"Final one_hot shape: {one_hot.shape}")
            # print(f"Final fg_labels length: {len(fg_labels)}")
            # print(f"Final bg_labels length: {len(bg_labels)}")

            # Normalize mixture audio
            logger.debug(f"MisophoniaDataset(__getitem__): Normalizing mixture audio")
            peak = torch.abs(mixture).max()
            if peak > 1:
                mixture /= peak
                target /= peak

            logger.debug(f"MisophoniaDataset(__getitem__): Creating inputs")
            inputs = {
                "mixture": mixture,
                "label_vector": one_hot,
            }

            if self.onflight_mode == 1:
                inputs["new_label_vector"] = one_hot.argmax(
                    dim=0
                )  # indexing for each label (currently including only one target)
                inputs["fg_labels"] = fg_labels
                inputs["bg_labels"] = bg_labels
                inputs["folder"] = idx

            logger.debug(f"MisophoniaDataset(__getitem__): Creating targets")
            targets = {
                "target": target,
                "num_target_speakers": len(fg_labels),
                "num_fg_labels": n_fg,
                "num_bg_labels": n_bg,
                "fg_labels": fg_labels,
                "bg_labels": bg_labels,
                "noise_labels": noise_labels,
                "num_noise_labels": n_noise,
                "total_labels": self.fg_sounds,
            }

            # print(f"Mixture shape in getitem: {mixture.shape}")
            # print(f"Target shape: {target.shape}")
            # print(f"dataloader __getitem__ took {time.time() - start:.3f}s")
            logger.debug(
                f"MisophoniaDataset(__getitem__): idx {idx}: took {time.time() - start:.3f}s"
            )
            logger.debug(
                f"MisophoniaDataset(__getitem__): Returning inputs and targets"
            )
            return inputs, targets

        except Exception as e:
            error_msg = f"Error in __getitem__ at index {idx}: {str(e)}"
            error_traceback = tb.format_exc()
            logging.error(error_msg)
            logging.error(f"Traceback:\n{error_traceback}")
            raise e  # Re-raise to stop training

    def save_dataset(self, path: str):
        test_loader = torch.utils.data.DataLoader(self, batch_size=1, shuffle=False)
        pbar = tqdm.tqdm(total=len(test_loader))
        for idx, dataset in enumerate(test_loader):
            inputs, targets = dataset
            sample_name = inputs["folder"]
            gt = targets["target"]
            mixture = inputs["mixture"]
            out_sample_dir = os.path.join(path, f"{idx:04d}")
            os.makedirs(out_sample_dir, exist_ok=True)
            utils.write_audio_file(
                os.path.join(out_sample_dir, "mixture.wav"),
                mixture.squeeze(0).numpy(),
                self.sr,
            )
            if isinstance(gt, torch.Tensor):
                utils.write_audio_file(
                    os.path.join(out_sample_dir, "gt.wav"),
                    gt.squeeze(0).numpy(),
                    self.sr,
                )

            metadata = {
                "file_path": out_sample_dir,
                "num_target_speakers": targets["num_target_speakers"],
                "target_labels": [label for label in inputs["fg_labels"] if label is not None],
                "total_labels": targets["total_labels"],
            }
            config = {
                "label_vector": inputs["label_vector"].tolist(),
                "num_target_speakers": targets["num_target_speakers"],
                "num_fg_labels": targets["num_fg_labels"],
                "num_bg_labels": targets["num_bg_labels"],
                "fg_labels": targets["fg_labels"],
                "bg_labels": targets["bg_labels"],
                "noise_labels": targets["noise_labels"],
                "num_noise_labels": targets["num_noise_labels"],
                "total_labels": targets["total_labels"],
            }
            with open(os.path.join(out_sample_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2, cls=TensorEncoder)
            with open(os.path.join(out_sample_dir, "config.json"), "w") as f:
                json.dump(config, f, indent=2, cls=TensorEncoder)
            pbar.update()
        logger.info(f"Saved {self.split} {len(self)} samples to {path}.")