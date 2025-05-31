import numpy as np
import scipy.signal
from scipy.fft import fft, fftfreq
from livekit import rtc
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import librosa  # For advanced audio analysis
import tracemalloc
import psutil
import threading
import gc
import time


@dataclass
class VoiceCharacteristics:
    """Comprehensive human voice analysis."""

    # Fundamental frequency analysis
    fundamental_freq: (
        float  # F0 (Hz) - human speech: 85-255 Hz (male), 165-265 Hz (female)
    )
    pitch_stability: float  # Variation in F0 (lower = more stable)

    # Formant analysis (speech-specific frequency bands)
    formant_f1: float  # First formant (vowel identification)
    formant_f2: float  # Second formant (vowel identification)
    formant_f3: float  # Third formant
    formant_clarity: float  # How distinct formants are

    # Voice quality indicators
    harmonics_to_noise_ratio: float  # HNR - speech has clear harmonics
    spectral_rolloff: float  # Frequency where 85% of spectral energy is below
    spectral_flux: float  # Rate of spectral change (speech has patterns)

    # Speech pattern analysis
    voiced_segments_ratio: float  # Ratio of voiced vs unvoiced speech
    speech_rate_estimate: float  # Syllables per second (human: 3-8)
    pause_pattern_score: float  # Natural speech has characteristic pauses

    # Voice uniqueness (anti-synthetic detection)
    micro_variations: (
        float  # Tiny pitch/amplitude variations (human voice has natural jitter)
    )
    breathing_artifacts: float  # Subtle breathing sounds between words
    vocal_tract_resonance: float  # Characteristic vocal tract filtering

    # Final assessment
    is_human_voice: bool  # Combined assessment
    confidence_score: float  # 0.0-1.0 confidence in human voice
    rejection_reasons: List[str]  # Why it might not be human voice


def analyze_voice_characteristics(
    frame: rtc.AudioFrame, context_frames: List[rtc.AudioFrame] = None
) -> VoiceCharacteristics:
    """Comprehensive human voice analysis."""

    # Convert to numpy array
    if isinstance(frame.data, memoryview):
        audio_data = np.frombuffer(frame.data, dtype=np.int16)
    else:
        audio_data = np.frombuffer(frame.data.tobytes(), dtype=np.int16)

    if len(audio_data) == 0:
        return VoiceCharacteristics(
            fundamental_freq=0,
            pitch_stability=0,
            formant_f1=0,
            formant_f2=0,
            formant_f3=0,
            formant_clarity=0,
            harmonics_to_noise_ratio=0,
            spectral_rolloff=0,
            spectral_flux=0,
            voiced_segments_ratio=0,
            speech_rate_estimate=0,
            pause_pattern_score=0,
            micro_variations=0,
            breathing_artifacts=0,
            vocal_tract_resonance=0,
            is_human_voice=False,
            confidence_score=0.0,
            rejection_reasons=["insufficient_data"],
        )

    # Convert to float for analysis
    audio_float = audio_data.astype(np.float32) / 32768.0
    sample_rate = frame.sample_rate

    rejection_reasons = []

    # 1. Fundamental frequency analysis (F0)
    fundamental_freq, pitch_stability = analyze_fundamental_frequency(
        audio_float, sample_rate
    )

    # Human speech F0 range check
    if fundamental_freq < 80 or fundamental_freq > 400:
        rejection_reasons.append(f"f0_out_of_range_{fundamental_freq:.1f}Hz")

    # 2. Formant analysis
    formants = analyze_formants(audio_float, sample_rate)
    formant_f1, formant_f2, formant_f3 = (
        formants[:3] if len(formants) >= 3 else (0, 0, 0)
    )
    formant_clarity = calculate_formant_clarity(audio_float, sample_rate, formants)

    # Human formant range check
    if not (200 <= formant_f1 <= 1000 and 800 <= formant_f2 <= 3000):
        rejection_reasons.append(
            f"formants_non_speech_f1_{formant_f1:.0f}_f2_{formant_f2:.0f}"
        )

    # 3. Harmonics-to-Noise Ratio
    harmonics_to_noise_ratio = calculate_hnr(audio_float, sample_rate, fundamental_freq)

    # Human speech typically has HNR > 10 dB
    if harmonics_to_noise_ratio < 5:
        rejection_reasons.append(f"low_hnr_{harmonics_to_noise_ratio:.1f}dB")

    # 4. Spectral analysis
    spectral_rolloff = calculate_spectral_rolloff(audio_float, sample_rate)
    spectral_flux = calculate_spectral_flux(audio_float, sample_rate)

    # Human speech energy concentrated below 4kHz
    if spectral_rolloff > 6000:
        rejection_reasons.append(f"high_spectral_rolloff_{spectral_rolloff:.0f}Hz")

    # 5. Voice activity and speech patterns
    voiced_segments_ratio = calculate_voiced_ratio(audio_float, sample_rate)
    speech_rate_estimate = estimate_speech_rate(audio_float, sample_rate)

    # 6. Natural voice variations
    micro_variations = analyze_micro_variations(audio_float, sample_rate)
    breathing_artifacts = detect_breathing_artifacts(audio_float, sample_rate)
    vocal_tract_resonance = analyze_vocal_tract_resonance(audio_float, sample_rate)

    # Synthetic voice often lacks micro-variations
    if micro_variations < 0.01:
        rejection_reasons.append("insufficient_micro_variations")

    # 7. Context analysis (if previous frames available)
    pause_pattern_score = 0.5  # Default
    if context_frames:
        pause_pattern_score = analyze_pause_patterns(context_frames)

    # 8. Final assessment
    confidence_factors = [
        (80 <= fundamental_freq <= 400, 0.2),  # F0 in human range
        (formant_clarity > 0.3, 0.15),  # Clear formants
        (harmonics_to_noise_ratio > 10, 0.15),  # Good HNR
        (200 <= formant_f1 <= 1000, 0.1),  # F1 in speech range
        (800 <= formant_f2 <= 3000, 0.1),  # F2 in speech range
        (spectral_rolloff < 4000, 0.1),  # Speech-like spectrum
        (voiced_segments_ratio > 0.3, 0.1),  # Sufficient voicing
        (micro_variations > 0.01, 0.05),  # Natural variations
        (breathing_artifacts > 0.02, 0.05),  # Natural breathing
    ]

    confidence_score = sum(
        weight for condition, weight in confidence_factors if condition
    )
    is_human_voice = confidence_score > 0.6 and len(rejection_reasons) == 0

    return VoiceCharacteristics(
        fundamental_freq=round(fundamental_freq, 1),
        pitch_stability=round(pitch_stability, 3),
        formant_f1=round(formant_f1, 1),
        formant_f2=round(formant_f2, 1),
        formant_f3=round(formant_f3, 1),
        formant_clarity=round(formant_clarity, 3),
        harmonics_to_noise_ratio=round(harmonics_to_noise_ratio, 1),
        spectral_rolloff=round(spectral_rolloff, 1),
        spectral_flux=round(spectral_flux, 3),
        voiced_segments_ratio=round(voiced_segments_ratio, 3),
        speech_rate_estimate=round(speech_rate_estimate, 1),
        pause_pattern_score=round(pause_pattern_score, 3),
        micro_variations=round(micro_variations, 4),
        breathing_artifacts=round(breathing_artifacts, 3),
        vocal_tract_resonance=round(vocal_tract_resonance, 3),
        is_human_voice=is_human_voice,
        confidence_score=round(confidence_score, 3),
        rejection_reasons=rejection_reasons,
    )


def analyze_fundamental_frequency(
    audio: np.ndarray, sample_rate: int
) -> Tuple[float, float]:
    """Analyze fundamental frequency (F0) and its stability."""
    # Use autocorrelation for F0 estimation
    autocorr = np.correlate(audio, audio, mode="full")
    autocorr = autocorr[len(autocorr) // 2 :]

    # Find peaks in autocorrelation
    min_period = int(sample_rate / 400)  # 400 Hz max
    max_period = int(sample_rate / 80)  # 80 Hz min

    if len(autocorr) <= max_period:
        return 0.0, 0.0

    autocorr_segment = autocorr[min_period:max_period]
    if len(autocorr_segment) == 0:
        return 0.0, 0.0

    peak_idx = np.argmax(autocorr_segment) + min_period
    fundamental_freq = sample_rate / peak_idx if peak_idx > 0 else 0.0

    # Calculate pitch stability (coefficient of variation)
    if len(audio) > sample_rate:  # At least 1 second
        # Analyze F0 over windows
        window_size = sample_rate // 10  # 100ms windows
        f0_estimates = []

        for i in range(0, len(audio) - window_size, window_size // 2):
            window = audio[i : i + window_size]
            f0_window, _ = analyze_fundamental_frequency(window, sample_rate)
            if f0_window > 0:
                f0_estimates.append(f0_window)

        if len(f0_estimates) > 1:
            pitch_stability = 1.0 - (np.std(f0_estimates) / np.mean(f0_estimates))
        else:
            pitch_stability = 0.0
    else:
        pitch_stability = 0.0

    return fundamental_freq, max(0.0, pitch_stability)


def analyze_formants(audio: np.ndarray, sample_rate: int) -> List[float]:
    """Analyze formant frequencies using LPC (Linear Predictive Coding)."""
    try:
        # Apply pre-emphasis filter
        pre_emphasized = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

        # LPC analysis (order 12-14 for formants)
        lpc_order = min(14, len(pre_emphasized) // 3)
        if lpc_order < 4:
            return [0, 0, 0]

        # Autocorrelation method for LPC
        autocorr = np.correlate(pre_emphasized, pre_emphasized, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]

        # Solve Yule-Walker equations
        if len(autocorr) <= lpc_order:
            return [0, 0, 0]

        r = autocorr[: lpc_order + 1]

        # Levinson-Durbin algorithm
        a = np.zeros(lpc_order + 1)
        a[0] = 1.0

        for m in range(1, lpc_order + 1):
            km = -np.sum(a[:m] * r[m:0:-1]) / r[0] if r[0] != 0 else 0
            a[m] = km
            for i in range(1, m):
                a[i] = a[i] + km * a[m - i]

        # Find formants from LPC roots
        roots = np.roots(a)

        # Extract formant frequencies
        formants = []
        for root in roots:
            if np.imag(root) > 0:  # Positive imaginary part
                freq = np.angle(root) * sample_rate / (2 * np.pi)
                if 100 < freq < sample_rate / 2:  # Valid frequency range
                    formants.append(freq)

        formants.sort()

        # Return first 3 formants
        while len(formants) < 3:
            formants.append(0)

        return formants[:3]

    except:
        return [0, 0, 0]


def calculate_formant_clarity(
    audio: np.ndarray, sample_rate: int, formants: List[float]
) -> float:
    """Calculate how distinct the formants are."""
    if not formants or all(f == 0 for f in formants):
        return 0.0

    # FFT to get spectrum
    fft_data = np.abs(fft(audio))
    freqs = fftfreq(len(audio), 1 / sample_rate)[: len(fft_data) // 2]
    spectrum = fft_data[: len(fft_data) // 2]

    clarity_scores = []

    for formant_freq in formants[:3]:
        if formant_freq == 0:
            continue

        # Find formant in spectrum
        formant_idx = np.argmin(np.abs(freqs - formant_freq))

        # Calculate peak prominence
        window_size = min(50, len(spectrum) // 20)
        start_idx = max(0, formant_idx - window_size)
        end_idx = min(len(spectrum), formant_idx + window_size)

        local_spectrum = spectrum[start_idx:end_idx]
        if len(local_spectrum) > 0:
            peak_value = spectrum[formant_idx]
            local_mean = np.mean(local_spectrum)
            if local_mean > 0:
                clarity = peak_value / local_mean
                clarity_scores.append(clarity)

    return np.mean(clarity_scores) if clarity_scores else 0.0


def calculate_hnr(audio: np.ndarray, sample_rate: int, f0: float) -> float:
    """Calculate Harmonics-to-Noise Ratio."""
    if f0 <= 0 or len(audio) < sample_rate // 10:
        return 0.0

    # FFT analysis
    fft_data = np.abs(fft(audio))
    freqs = fftfreq(len(audio), 1 / sample_rate)[: len(fft_data) // 2]
    spectrum = fft_data[: len(fft_data) // 2]

    # Find harmonic peaks
    harmonic_energy = 0
    noise_energy = 0

    for harmonic in range(1, 6):  # First 5 harmonics
        harmonic_freq = f0 * harmonic
        if harmonic_freq >= sample_rate / 2:
            break

        # Find harmonic peak
        harmonic_idx = np.argmin(np.abs(freqs - harmonic_freq))

        # Define harmonic band (±5% of fundamental)
        band_width = max(1, int(0.05 * f0 * len(freqs) / (sample_rate / 2)))
        start_idx = max(0, harmonic_idx - band_width)
        end_idx = min(len(spectrum), harmonic_idx + band_width)

        harmonic_band = spectrum[start_idx:end_idx]
        harmonic_energy += np.max(harmonic_band) ** 2

        # Noise estimate (surrounding regions)
        noise_start1 = max(0, start_idx - band_width * 2)
        noise_end1 = start_idx
        noise_start2 = end_idx
        noise_end2 = min(len(spectrum), end_idx + band_width * 2)

        noise_band1 = spectrum[noise_start1:noise_end1]
        noise_band2 = spectrum[noise_start2:noise_end2]

        if len(noise_band1) > 0:
            noise_energy += np.mean(noise_band1**2) * len(harmonic_band)
        if len(noise_band2) > 0:
            noise_energy += np.mean(noise_band2**2) * len(harmonic_band)

    if noise_energy > 0:
        hnr_linear = harmonic_energy / noise_energy
        hnr_db = 10 * np.log10(max(hnr_linear, 1e-10))
        return hnr_db
    else:
        return 0.0


def analyze_micro_variations(audio: np.ndarray, sample_rate: int) -> float:
    """Analyze micro-variations in amplitude and frequency (jitter/shimmer)."""
    if len(audio) < sample_rate // 10:  # Need at least 100ms
        return 0.0

    # Calculate short-term amplitude variations
    window_size = sample_rate // 100  # 10ms windows
    amplitudes = []

    for i in range(0, len(audio) - window_size, window_size):
        window = audio[i : i + window_size]
        amplitude = np.sqrt(np.mean(window**2))
        amplitudes.append(amplitude)

    if len(amplitudes) < 2:
        return 0.0

    # Shimmer: amplitude variation
    amplitude_variation = (
        np.std(amplitudes) / np.mean(amplitudes) if np.mean(amplitudes) > 0 else 0
    )

    # Frequency variation using zero-crossing rate
    zcr_windows = []
    for i in range(0, len(audio) - window_size, window_size):
        window = audio[i : i + window_size]
        zero_crossings = np.where(np.diff(np.signbit(window)))[0]
        zcr = len(zero_crossings) / len(window)
        zcr_windows.append(zcr)

    frequency_variation = (
        np.std(zcr_windows) / np.mean(zcr_windows) if np.mean(zcr_windows) > 0 else 0
    )

    # Combined micro-variation score
    return (amplitude_variation + frequency_variation) / 2


def detect_breathing_artifacts(audio: np.ndarray, sample_rate: int) -> float:
    """Detect breathing artifacts (low-frequency, low-amplitude sounds)."""
    # Low-pass filter to isolate breathing frequencies (< 500 Hz)
    nyquist = sample_rate / 2
    low_cutoff = 500 / nyquist

    if low_cutoff >= 1.0:
        return 0.0

    try:
        from scipy import signal

        b, a = signal.butter(4, low_cutoff, btype="low")
        low_freq_audio = signal.filtfilt(b, a, audio)

        # Breathing is typically very low amplitude but present
        breathing_energy = np.sqrt(np.mean(low_freq_audio**2))

        # Normalize by total energy
        total_energy = np.sqrt(np.mean(audio**2))
        if total_energy > 0:
            breathing_ratio = breathing_energy / total_energy
            return min(breathing_ratio, 1.0)
        else:
            return 0.0
    except:
        return 0.0


# Additional helper functions...
def calculate_spectral_rolloff(
    audio: np.ndarray, sample_rate: int, threshold: float = 0.85
) -> float:
    """Calculate spectral rolloff (frequency below which 85% of energy lies)."""
    fft_data = np.abs(fft(audio))
    spectrum = fft_data[: len(fft_data) // 2]
    freqs = fftfreq(len(audio), 1 / sample_rate)[: len(spectrum)]

    total_energy = np.sum(spectrum**2)
    if total_energy == 0:
        return 0.0

    cumulative_energy = np.cumsum(spectrum**2)
    rolloff_idx = np.where(cumulative_energy >= threshold * total_energy)[0]

    if len(rolloff_idx) > 0:
        return freqs[rolloff_idx[0]]
    else:
        return freqs[-1]


def calculate_spectral_flux(audio: np.ndarray, sample_rate: int) -> float:
    """Calculate spectral flux (rate of spectral change)."""
    window_size = sample_rate // 20  # 50ms windows
    if len(audio) < window_size * 2:
        return 0.0

    spectral_changes = []
    prev_spectrum = None

    for i in range(0, len(audio) - window_size, window_size // 2):
        window = audio[i : i + window_size]
        spectrum = np.abs(fft(window))[: window_size // 2]

        if prev_spectrum is not None and len(spectrum) == len(prev_spectrum):
            # Calculate spectral difference
            diff = np.sum((spectrum - prev_spectrum) ** 2)
            spectral_changes.append(diff)

        prev_spectrum = spectrum

    return np.mean(spectral_changes) if spectral_changes else 0.0


def calculate_voiced_ratio(audio: np.ndarray, sample_rate: int) -> float:
    """Calculate ratio of voiced segments in audio."""
    window_size = sample_rate // 20  # 50ms windows
    voiced_windows = 0
    total_windows = 0

    for i in range(0, len(audio) - window_size, window_size // 2):
        window = audio[i : i + window_size]

        # Simple voicing detection using autocorrelation
        autocorr = np.correlate(window, window, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]

        if len(autocorr) > 10:
            # Look for periodicity
            max_autocorr = np.max(autocorr[1 : len(autocorr) // 4])
            if max_autocorr > 0.3 * autocorr[0]:  # Threshold for voicing
                voiced_windows += 1

        total_windows += 1

    return voiced_windows / total_windows if total_windows > 0 else 0.0


def estimate_speech_rate(audio: np.ndarray, sample_rate: int) -> float:
    """Estimate speech rate (syllables per second)."""
    # Simplified syllable detection using amplitude peaks
    # Smooth the signal
    window_size = sample_rate // 50  # 20ms smoothing
    smoothed = np.convolve(
        np.abs(audio), np.ones(window_size) / window_size, mode="same"
    )

    # Find peaks (potential syllables)
    from scipy.signal import find_peaks

    peaks, _ = find_peaks(
        smoothed, height=np.max(smoothed) * 0.3, distance=sample_rate // 10
    )

    duration_seconds = len(audio) / sample_rate
    if duration_seconds > 0:
        return len(peaks) / duration_seconds
    else:
        return 0.0


def analyze_vocal_tract_resonance(audio: np.ndarray, sample_rate: int) -> float:
    """Analyze vocal tract resonance characteristics."""
    # Look for characteristic vocal tract filtering
    fft_data = np.abs(fft(audio))
    spectrum = fft_data[: len(fft_data) // 2]
    freqs = fftfreq(len(audio), 1 / sample_rate)[: len(spectrum)]

    # Human vocal tract creates specific resonance patterns
    # Look for energy distribution in speech bands
    speech_bands = [
        (300, 800),  # Low formant region
        (800, 2000),  # Mid formant region
        (2000, 4000),  # High formant region
    ]

    band_energies = []
    for low, high in speech_bands:
        mask = (freqs >= low) & (freqs <= high)
        band_energy = np.sum(spectrum[mask] ** 2)
        band_energies.append(band_energy)

    total_energy = np.sum(spectrum**2)
    if total_energy > 0:
        speech_energy_ratio = sum(band_energies) / total_energy
        return min(speech_energy_ratio, 1.0)
    else:
        return 0.0


def analyze_pause_patterns(frames: List[rtc.AudioFrame]) -> float:
    """Analyze pause patterns across multiple frames."""
    if len(frames) < 5:
        return 0.5

    # Convert frames to continuous audio
    audio_segments = []
    for frame in frames:
        if isinstance(frame.data, memoryview):
            audio_data = np.frombuffer(frame.data, dtype=np.int16)
        else:
            audio_data = np.frombuffer(frame.data.tobytes(), dtype=np.int16)
        audio_segments.append(audio_data.astype(np.float32) / 32768.0)

    full_audio = np.concatenate(audio_segments)
    sample_rate = frames[0].sample_rate

    # Detect speech/pause segments
    window_size = sample_rate // 20  # 50ms windows
    energy_threshold = 0.01

    speech_segments = []
    pause_segments = []

    for i in range(0, len(full_audio) - window_size, window_size):
        window = full_audio[i : i + window_size]
        energy = np.sqrt(np.mean(window**2))

        if energy > energy_threshold:
            speech_segments.append(len(speech_segments))
        else:
            pause_segments.append(len(pause_segments))

    # Analyze pause pattern naturalness
    if len(speech_segments) + len(pause_segments) == 0:
        return 0.0

    pause_ratio = len(pause_segments) / (len(speech_segments) + len(pause_segments))

    # Natural speech has 20-40% pauses
    if 0.2 <= pause_ratio <= 0.4:
        return 1.0
    elif 0.1 <= pause_ratio <= 0.6:
        return 0.7
    else:
        return 0.3


class MicroscopicProfiler:
    """Ultra-detailed performance and memory profiling."""

    def __init__(self):
        self._memory_snapshots = {}
        self._cpu_samples = []
        self._thread_switches = []
        self._gc_collections = []
        tracemalloc.start()

    def snapshot_memory(self, location: str, frame_id: str) -> Dict[str, Any]:
        """Take detailed memory snapshot."""
        current, peak = tracemalloc.get_traced_memory()
        snapshot = tracemalloc.take_snapshot()

        # Get top memory allocations
        top_stats = snapshot.statistics("lineno")[:5]
        allocations = []
        for stat in top_stats:
            allocations.append(
                {
                    "filename": stat.traceback.format()[-1],
                    "size_mb": stat.size / 1024 / 1024,
                    "count": stat.count,
                }
            )

        process = psutil.Process()
        memory_info = process.memory_info()

        memory_data = {
            "location": location,
            "frame_id": frame_id,
            "traced_current_mb": current / 1024 / 1024,
            "traced_peak_mb": peak / 1024 / 1024,
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent(),
            "top_allocations": allocations,
            "gc_counts": gc.get_count(),
            "timestamp_ns": time.time_ns(),
        }

        return memory_data

    def profile_cpu_usage(self, location: str, frame_id: str) -> Dict[str, Any]:
        """Profile CPU usage and thread information."""
        process = psutil.Process()

        cpu_data = {
            "location": location,
            "frame_id": frame_id,
            "cpu_percent": process.cpu_percent(),
            "num_threads": process.num_threads(),
            "thread_id": threading.get_ident(),
            "context_switches": process.num_ctx_switches()._asdict(),
            "timestamp_ns": time.time_ns(),
        }

        return cpu_data


# Global profiler instance
_profiler = MicroscopicProfiler()
