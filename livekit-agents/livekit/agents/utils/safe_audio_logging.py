import time
import threading
import asyncio
from contextlib import contextmanager
from livekit import rtc
from livekit.agents.log import logger
from typing import Optional, Dict, Any, List
import weakref
from dataclasses import dataclass

import numpy as np
import math


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
    try:
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
    except Exception as e:
        # Safe fallback on any error
        return SimpleAudioMetrics(
            dbfs=-float("inf"),
            rms_energy=0.0,
            peak_amplitude=0.0,
            zero_crossing_rate=0.0,
            spectral_centroid=0.0,
            has_audio_content=False,
        )


class SafeAudioLogger:
    """Ultra-safe audio logging that cannot crash or impact original functionality."""

    def __init__(self, enabled: bool = True, max_memory_mb: int = 50):
        self.enabled = enabled
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.current_memory_estimate = 0
        self._session_start = time.time_ns()
        self._frame_counter = 0
        self._lock = threading.Lock()

        # Use WeakValueDictionary to prevent memory leaks
        self._frame_tracking = weakref.WeakValueDictionary()

        # Frame tracking for final statistics
        self._frame_checkpoints: Dict[str, List[Dict]] = {}
        self._frame_checkpoints_lock = threading.Lock()

    def safe_execute(self, func, *args, **kwargs):
        """Execute function safely, never allowing exceptions to propagate."""
        if not self.enabled:
            return None

        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Log the logging error, but never let it crash the main code
            try:
                logger.warning(f"Audio logging error (non-critical): {e}")
            except:
                pass  # Even logging the error failed, but we continue
            return None

    def get_frame_id(self) -> str:
        """Generate frame ID safely."""
        try:
            with self._lock:
                self._frame_counter += 1
                return f"f_{self._frame_counter}_{int(time.time() * 1000000) % 1000000}"
        except:
            # Fallback if even this fails
            return f"fallback_{int(time.time() * 1000)}"

    def basic_frame_info(self, frame: rtc.AudioFrame) -> Dict[str, Any]:
        """Extract only basic, safe frame information."""
        try:
            return {
                "duration_ms": round(frame.duration * 1000, 2),
                "sample_rate": frame.sample_rate,
                "num_channels": frame.num_channels,
                "samples_per_channel": frame.samples_per_channel,
                "data_size_bytes": len(frame.data) if frame.data else 0,
                "timestamp_ns": time.time_ns(),
                "session_relative_ns": time.time_ns() - self._session_start,
            }
        except:
            return {
                "error": "failed_to_extract_basic_info",
                "timestamp_ns": time.time_ns(),
            }

    def simple_audio_content_check(self, frame: rtc.AudioFrame) -> bool:
        """Check if frame contains audio content based on RMS > 0 (dBFS > -inf)."""
        try:
            if not frame or not frame.data or len(frame.data) == 0:
                return False

            # Convert frame data to numpy array
            if isinstance(frame.data, memoryview):
                audio_data = np.frombuffer(frame.data, dtype=np.int16)
            else:
                # Attempt to convert to bytes if it's not already, then frombuffer
                # This handles cases where frame.data might be a different buffer type
                audio_data = np.frombuffer(
                    (
                        frame.data.tobytes()
                        if hasattr(frame.data, "tobytes")
                        else bytes(frame.data)
                    ),
                    dtype=np.int16,
                )

            if len(audio_data) == 0:
                return False  # Should have been caught by frame.data check, but as a safeguard

            # Convert to float64 for precision in RMS calculation
            audio_float = audio_data.astype(np.float64)

            # Calculate RMS
            # np.mean will be NaN for empty array, but we check len(audio_data) == 0 earlier
            rms = np.sqrt(np.mean(np.square(audio_float)))

            # If RMS > 0, then dBFS > -infinity, meaning there's some audio content
            has_content = bool(rms > 0)
            if has_content:
                logger.info(f"DBFS: {20 * np.log10(rms)}")
            return has_content

        except Exception as e:
            # If any calculation fails, log the error and assume it has content (safe fallback)
            try:
                logger.warning(
                    f"Audio content check error (falling back to True): {e}",
                    exc_info=True,
                )
            except:
                pass  # Even logging the error failed
            return True

    def track_frame_checkpoint(self, frame_id: str, location: str, timestamp_ns: int):
        """Track checkpoints for each frame to generate final stats."""
        if not frame_id:
            return

        try:
            with self._frame_checkpoints_lock:
                if frame_id not in self._frame_checkpoints:
                    self._frame_checkpoints[frame_id] = []
                self._frame_checkpoints[frame_id].append(
                    {
                        "location": location,
                        "timestamp_ns": timestamp_ns,
                        "timestamp_ms": timestamp_ns / 1_000_000,
                    }
                )
        except:
            pass  # Safe fallback

    def log_frame_final_stats(self, frame_id: str):
        """Log comprehensive final statistics for a frame after it completes the pipeline."""
        if not frame_id:
            return

        try:
            with self._frame_checkpoints_lock:
                if frame_id not in self._frame_checkpoints:
                    return

                checkpoints = sorted(
                    self._frame_checkpoints[frame_id], key=lambda x: x["timestamp_ns"]
                )

                if len(checkpoints) < 2:
                    return

                # Calculate timing between checkpoints
                timing_stats = {
                    "event": "FRAME_FINAL_STATS",
                    "frame_id": frame_id,
                    "total_pipeline_duration_ms": round(
                        (
                            checkpoints[-1]["timestamp_ns"]
                            - checkpoints[0]["timestamp_ns"]
                        )
                        / 1_000_000,
                        3,
                    ),
                    "first_checkpoint": checkpoints[0]["location"],
                    "last_checkpoint": checkpoints[-1]["location"],
                    "checkpoint_count": len(checkpoints),
                    "checkpoints_timeline": [],
                }

                # Add detailed checkpoint timeline
                for i, checkpoint in enumerate(checkpoints):
                    timeline_entry = {
                        "sequence": i + 1,
                        "location": checkpoint["location"],
                        "absolute_timestamp_ns": checkpoint["timestamp_ns"],
                        "relative_time_from_start_ms": round(
                            (
                                checkpoint["timestamp_ns"]
                                - checkpoints[0]["timestamp_ns"]
                            )
                            / 1_000_000,
                            3,
                        ),
                    }

                    # Add time since previous checkpoint
                    if i > 0:
                        timeline_entry["time_since_previous_ms"] = round(
                            (
                                checkpoint["timestamp_ns"]
                                - checkpoints[i - 1]["timestamp_ns"]
                            )
                            / 1_000_000,
                            3,
                        )

                    timing_stats["checkpoints_timeline"].append(timeline_entry)

                # Calculate latency between key checkpoints
                checkpoint_latencies = {}
                checkpoint_map = {
                    cp["location"]: cp["timestamp_ns"] for cp in checkpoints
                }

                key_checkpoints = [
                    "webrtc_input",
                    "audio_recognition_input",
                    "vad_frame_dequeue",
                    "vad_inference_complete",
                    "audiobytestream_input",
                ]

                for i in range(len(key_checkpoints) - 1):
                    from_checkpoint = key_checkpoints[i]
                    to_checkpoint = key_checkpoints[i + 1]

                    if (
                        from_checkpoint in checkpoint_map
                        and to_checkpoint in checkpoint_map
                    ):
                        latency_ms = round(
                            (
                                checkpoint_map[to_checkpoint]
                                - checkpoint_map[from_checkpoint]
                            )
                            / 1_000_000,
                            3,
                        )
                        checkpoint_latencies[
                            f"{from_checkpoint}_to_{to_checkpoint}_ms"
                        ] = latency_ms

                timing_stats["checkpoint_latencies"] = checkpoint_latencies

                # logger.info(f"COMPREHENSIVE_AUDIO_LOG", extra=timing_stats)
                logger.info(f"COMPREHENSIVE_AUDIO_LOG: {timing_stats}")

                # Clean up to prevent memory leaks
                del self._frame_checkpoints[frame_id]
        except:
            pass  # Safe fallback

    def cleanup_old_frame_tracking(self):
        """Clean up old frame tracking data to prevent memory leaks."""
        try:
            current_time = time.time_ns()
            cutoff_time = current_time - (30 * 1_000_000_000)  # 30 seconds ago

            with self._frame_checkpoints_lock:
                frames_to_remove = []
                for frame_id, checkpoints in self._frame_checkpoints.items():
                    if checkpoints and checkpoints[-1]["timestamp_ns"] < cutoff_time:
                        frames_to_remove.append(frame_id)

                for frame_id in frames_to_remove:
                    del self._frame_checkpoints[frame_id]
        except:
            pass

    def log_checkpoint(
        self,
        location: str,
        frame: rtc.AudioFrame = None,
        frame_id: str = None,
        extra: Dict = None,
        filter_silent: bool = False,
        calculate_metrics: bool = False,
    ) -> Optional[str]:
        """Log a checkpoint safely with optional comprehensive metrics."""
        return self.safe_execute(
            self._log_checkpoint_impl,
            location,
            frame,
            frame_id,
            extra,
            filter_silent,
            calculate_metrics,
        )

    def _log_checkpoint_impl(
        self,
        location: str,
        frame: rtc.AudioFrame = None,
        frame_id: str = None,
        extra: Dict = None,
        filter_silent: bool = False,
        calculate_metrics: bool = False,
    ) -> Optional[str]:
        """Internal implementation of checkpoint logging."""

        # Memory protection
        if self.current_memory_estimate > self.max_memory_bytes:
            return frame_id  # Skip logging to prevent memory issues

        # Optional: Filter out silent frames from being logged for this specific checkpoint
        if filter_silent and frame is not None:
            if not self.simple_audio_content_check(frame):
                return frame_id  # Return the (potentially newly generated) frame_id, but don't log *this* checkpoint.

        if frame_id is None:
            frame_id = self.get_frame_id()

        current_time_ns = time.time_ns()

        log_data = {
            "event": "AUDIO_FLOW" if calculate_metrics else "AUDIO_CHECKPOINT",
            "location": location,
            "frame_id": frame_id,
            "timestamp_ns": current_time_ns,
        }

        if frame is not None:
            # Only extract basic info, no complex calculations
            frame_info = self.basic_frame_info(frame)
            log_data.update(frame_info)

            # Add simple audio content info if available
            if filter_silent:
                log_data["has_audio_content"] = self.simple_audio_content_check(frame)

            # Add comprehensive metrics if requested
            if calculate_metrics:
                metrics = calculate_audio_metrics(frame)
                log_data.update(
                    {
                        "dbfs": metrics.dbfs,
                        "rms_energy": metrics.rms_energy,
                        "peak_amplitude": metrics.peak_amplitude,
                        "zero_crossing_rate": metrics.zero_crossing_rate,
                        "spectral_centroid": metrics.spectral_centroid,
                        "comprehensive_has_audio_content": metrics.has_audio_content,
                    }
                )

        if extra:
            # Safely merge extra data
            try:
                log_data.update(extra)
            except:
                log_data["extra_data_error"] = True

        # Estimate memory usage (very rough)
        self.current_memory_estimate += len(str(log_data))

        # logger.info("COMPREHENSIVE_AUDIO_LOG", extra=log_data)
        logger.info(f"COMPREHENSIVE_AUDIO_LOG: {log_data}")

        # Track frame for final stats
        self.track_frame_checkpoint(frame_id, location, current_time_ns)

        return frame_id

    def log_timing(
        self, frame_id: str, operation: str, duration_ns: int, extra: Dict = None
    ):
        """Log timing information safely."""
        return self.safe_execute(
            self._log_timing_impl, frame_id, operation, duration_ns, extra
        )

    def _log_timing_impl(
        self, frame_id: str, operation: str, duration_ns: int, extra: Dict = None
    ):
        """Internal timing logging implementation."""
        if frame_id is None:
            return

        log_data = {
            "event": "PROCESSING_STEP",
            "frame_id": frame_id,
            "operation": operation,
            "duration_ns": duration_ns,
            "duration_ms": round(duration_ns / 1_000_000, 3),
            "timestamp_ns": time.time_ns(),
        }

        if extra:
            try:
                log_data.update(extra)
            except:
                log_data["extra_data_error"] = True

        # logger.info("COMPREHENSIVE_AUDIO_LOG", extra=log_data)
        logger.info(f"COMPREHENSIVE_AUDIO_LOG: {log_data}")

    def cleanup_old_data(self):
        """Clean up old tracking data safely."""
        try:
            # Reset memory estimate periodically
            if self.current_memory_estimate > self.max_memory_bytes // 2:
                self.current_memory_estimate = 0

            # Clean up old frame tracking
            self.cleanup_old_frame_tracking()
        except:
            pass


# Global safe logger instance
_safe_logger = SafeAudioLogger()


@contextmanager
def safe_timing(frame_id: str, operation: str, extra: Dict = None):
    """Context manager for safe timing measurement."""
    start_time = time.time_ns()
    try:
        yield
    finally:
        end_time = time.time_ns()
        _safe_logger.log_timing(frame_id, operation, end_time - start_time, extra)


def safe_log_checkpoint(
    location: str,
    frame: rtc.AudioFrame = None,
    frame_id: str = None,
    extra: Dict = None,
    filter_silent: bool = False,
    calculate_metrics: bool = False,
) -> Optional[str]:
    """Safe checkpoint logging function with optional comprehensive metrics."""
    return _safe_logger.log_checkpoint(
        location, frame, frame_id, extra, filter_silent, calculate_metrics
    )


def safe_log_timing(
    frame_id: str, operation: str, duration_ns: int, extra: Dict = None
):
    """Safe timing logging function."""
    return _safe_logger.log_timing(frame_id, operation, duration_ns, extra)


def configure_safe_logging(enabled: bool = True, max_memory_mb: int = 50):
    """Configure safe logging settings."""
    global _safe_logger
    _safe_logger.enabled = enabled
    _safe_logger.max_memory_bytes = max_memory_mb * 1024 * 1024


def log_frame_final_stats(frame_id: str):
    """Generate final stats for a frame."""
    _safe_logger.log_frame_final_stats(frame_id)
