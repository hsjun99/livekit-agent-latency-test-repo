import time
import threading
import asyncio
from contextlib import contextmanager
from livekit import rtc
from livekit.agents.log import logger
from typing import Optional, Dict, Any
import weakref

import numpy as np
import math


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

            # logger.info(f"RMS: {rms}")

            # has_content = rms > 0
            has_content = 20 * np.log10(rms) > 30
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

    def log_checkpoint(
        self,
        location: str,
        frame: rtc.AudioFrame = None,
        frame_id: str = None,
        extra: Dict = None,
        filter_silent: bool = False,
    ) -> Optional[str]:
        """Log a checkpoint safely."""
        return self.safe_execute(
            self._log_checkpoint_impl, location, frame, frame_id, extra, filter_silent
        )

    def _log_checkpoint_impl(
        self,
        location: str,
        frame: rtc.AudioFrame = None,
        frame_id: str = None,
        extra: Dict = None,
        filter_silent: bool = False,
    ) -> Optional[str]:
        """Internal implementation of checkpoint logging."""

        # Memory protection
        if self.current_memory_estimate > self.max_memory_bytes:
            return frame_id  # Skip logging to prevent memory issues

        # Optional: Filter out silent frames
        if filter_silent and frame is not None:
            if not self.simple_audio_content_check(frame):
                return frame_id  # Skip logging silent frames

        if frame_id is None:
            frame_id = self.get_frame_id()

        log_data = {
            "event": "AUDIO_CHECKPOINT",
            "location": location,
            "frame_id": frame_id,
            "timestamp_ns": time.time_ns(),
        }

        if frame is not None:
            # Only extract basic info, no complex calculations
            frame_info = self.basic_frame_info(frame)
            log_data.update(frame_info)

            # Add simple audio content info if available
            if filter_silent:
                log_data["has_audio_content"] = self.simple_audio_content_check(frame)

        if extra:
            # Safely merge extra data
            try:
                log_data.update(extra)
            except:
                log_data["extra_data_error"] = True

        # Estimate memory usage (very rough)
        self.current_memory_estimate += len(str(log_data))

        # logger.info("SAFE_AUDIO_LOG", extra=log_data)
        logger.info(f"SAFE_AUDIO_LOG: {log_data}")
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
            "event": "AUDIO_TIMING",
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

        # logger.info("SAFE_AUDIO_LOG", extra=log_data)
        logger.info(f"SAFE_AUDIO_LOG: {log_data}")

    def cleanup_old_data(self):
        """Clean up old tracking data safely."""
        try:
            # Reset memory estimate periodically
            if self.current_memory_estimate > self.max_memory_bytes // 2:
                self.current_memory_estimate = 0
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
) -> Optional[str]:
    """Safe checkpoint logging function."""
    return _safe_logger.log_checkpoint(location, frame, frame_id, extra, filter_silent)


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
