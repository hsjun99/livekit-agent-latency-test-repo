import time
import numpy as np
import math
import threading
from livekit import rtc
from livekit.agents.log import logger
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class SimpleAudioMetrics:
    """Simple but comprehensive audio analysis."""

    dbfs: float  # dBFS (decibels relative to full scale)
    rms_energy: float  # Root Mean Square energy
    peak_amplitude: float  # Peak amplitude (0.0-1.0)
    zero_crossing_rate: float  # Zero crossing rate
    spectral_centroid: float  # Simple frequency content measure
    has_audio_content: bool  # Whether frame contains meaningful audio


def calculate_audio_metrics(frame: rtc.AudioFrame) -> SimpleAudioMetrics:
    """Calculate comprehensive audio metrics using only numpy."""

    # Convert frame data to numpy array
    if isinstance(frame.data, memoryview):
        audio_data = np.frombuffer(frame.data, dtype=np.int16)
    else:
        audio_data = np.frombuffer(frame.data.tobytes(), dtype=np.int16)

    if len(audio_data) == 0:
        return SimpleAudioMetrics(
            dbfs=-float("inf"),
            rms_energy=0.0,
            peak_amplitude=0.0,
            zero_crossing_rate=0.0,
            spectral_centroid=0.0,
            has_audio_content=False,
        )

    # Convert to float64 for precision
    audio_float = audio_data.astype(np.float64)

    # 1. dBFS calculation
    rms = np.sqrt(np.mean(np.square(audio_float)))
    if rms == 0:
        dbfs = -float("inf")
    else:
        dbfs = 20 * math.log10(rms / 32767.0)

    # 2. RMS Energy (normalized)
    rms_energy = rms / 32767.0

    # 3. Peak amplitude
    peak_amplitude = np.max(np.abs(audio_float)) / 32767.0

    # 4. Zero Crossing Rate
    zero_crossings = np.where(np.diff(np.signbit(audio_float)))[0]
    zero_crossing_rate = (
        len(zero_crossings) / len(audio_float) if len(audio_float) > 1 else 0.0
    )

    # 5. Simple spectral centroid using FFT
    if len(audio_float) >= 64:  # Minimum size for meaningful FFT
        fft_data = np.fft.rfft(audio_float)
        magnitude = np.abs(fft_data)
        freqs = np.fft.rfftfreq(len(audio_float), 1.0 / frame.sample_rate)

        # Spectral centroid (weighted average frequency)
        if np.sum(magnitude) > 0:
            spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        else:
            spectral_centroid = 0.0
    else:
        spectral_centroid = 0.0

    # 6. Simple audio content detection
    has_audio_content = (
        dbfs > -50.0  # Sufficient volume
        and rms_energy > 0.001  # Minimum energy
        and zero_crossing_rate > 0.01  # Some frequency content
    )

    return SimpleAudioMetrics(
        dbfs=round(dbfs, 2),
        rms_energy=round(rms_energy, 4),
        peak_amplitude=round(peak_amplitude, 4),
        zero_crossing_rate=round(zero_crossing_rate, 4),
        spectral_centroid=round(spectral_centroid, 1),
        has_audio_content=has_audio_content,
    )


# Global session tracking
_session_start_time = time.time_ns()
_frame_counter = 0
_frame_counter_lock = threading.Lock()


def get_precise_timing() -> Dict[str, int]:
    """Get precise timing information."""
    current_time = time.time_ns()
    return {
        "timestamp_ns": current_time,
        "session_relative_ns": current_time - _session_start_time,
        "timestamp_us": current_time // 1000,
        "timestamp_ms": current_time // 1_000_000,
    }


def generate_frame_id() -> str:
    """Generate unique frame ID for tracking."""
    global _frame_counter
    with _frame_counter_lock:
        _frame_counter += 1
        return f"frame_{_frame_counter:08d}_{int(time.time() * 1000000)}"


def should_log_frame(metrics: SimpleAudioMetrics, location: str) -> bool:
    """Simple logic to determine if frame should be logged."""
    # Log if has audio content OR at critical pipeline points
    return metrics.has_audio_content or location in [
        "webrtc_input",
        "audio_recognition_input",
        "vad_inference_complete",
    ]


def log_audio_frame(
    frame: rtc.AudioFrame, location: str, frame_id: str = None, extra_data: dict = None
) -> Optional[str]:
    """Simple but comprehensive audio frame logging."""

    # Calculate audio metrics
    metrics = calculate_audio_metrics(frame)

    # Check if we should log this frame
    if not should_log_frame(metrics, location):
        return frame_id

    # Generate or use provided frame ID
    if frame_id is None:
        frame_id = generate_frame_id()

    # Get precise timing
    timing = get_precise_timing()

    # Prepare log data
    log_data = {
        "location": location,
        "frame_id": frame_id,
        "duration_ms": round(frame.duration * 1000, 2),
        "sample_rate": frame.sample_rate,
        "num_channels": frame.num_channels,
        "samples_per_channel": frame.samples_per_channel,
        "data_size_bytes": len(frame.data),
        # Timing information
        **timing,
        # Audio metrics
        "dbfs": metrics.dbfs,
        "rms_energy": metrics.rms_energy,
        "peak_amplitude": metrics.peak_amplitude,
        "zero_crossing_rate": metrics.zero_crossing_rate,
        "spectral_centroid": metrics.spectral_centroid,
        "has_audio_content": metrics.has_audio_content,
    }

    if extra_data:
        log_data.update(extra_data)

    logger.info("AUDIO_FLOW", extra=log_data)
    return frame_id


def log_processing_step(
    frame_id: str,
    location: str,
    operation: str,
    start_time_ns: int,
    end_time_ns: int,
    extra_data: dict = None,
):
    """Log a processing step with precise timing."""
    processing_time_ms = (end_time_ns - start_time_ns) / 1_000_000
    timing = get_precise_timing()

    log_data = {
        "frame_id": frame_id,
        "location": location,
        "operation": operation,
        "processing_time_ms": round(processing_time_ms, 3),
        "start_time_ns": start_time_ns,
        "end_time_ns": end_time_ns,
        **timing,
    }

    if extra_data:
        log_data.update(extra_data)

    logger.info("PROCESSING_STEP", extra=log_data)


def log_memory_operation(
    frame_id: str, location: str, operation: str, size_bytes: int, duration_ns: int
):
    """Log memory operations."""
    timing = get_precise_timing()

    log_data = {
        "frame_id": frame_id,
        "location": location,
        "operation": operation,
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / 1024 / 1024, 3),
        "duration_ns": duration_ns,
        "duration_ms": round(duration_ns / 1_000_000, 3),
        **timing,
    }

    logger.info("MEMORY_OPERATION", extra=log_data)


def log_channel_operation(
    frame_id: str,
    location: str,
    operation: str,
    duration_ns: int,
    queue_size: int = None,
):
    """Log async channel operations."""
    timing = get_precise_timing()

    log_data = {
        "frame_id": frame_id,
        "location": location,
        "operation": operation,
        "duration_ns": duration_ns,
        "duration_ms": round(duration_ns / 1_000_000, 3),
        **timing,
    }

    if queue_size is not None:
        log_data["queue_size"] = queue_size

    logger.info("CHANNEL_OPERATION", extra=log_data)
