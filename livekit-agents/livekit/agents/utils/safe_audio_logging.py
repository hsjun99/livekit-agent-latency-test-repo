import time
import threading
import asyncio
from contextlib import contextmanager
from livekit import rtc
from livekit.agents.log import logger
from typing import Optional, Dict, Any
import weakref


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
        """Simple, safe check if frame contains audio content (no complex calculations)."""
        try:
            # Basic checks that are very fast and safe
            if not frame.data or len(frame.data) == 0:
                return False

            # Convert to bytes if needed (safe operation)
            if hasattr(frame.data, "tobytes"):
                data_bytes = frame.data.tobytes()
            else:
                data_bytes = bytes(frame.data)

            # Simple check: if most bytes are zero, likely silence
            # This is much faster and safer than FFT/RMS calculations
            zero_count = data_bytes.count(b"\x00")
            silence_ratio = zero_count / len(data_bytes)

            # If more than 95% of bytes are zero, consider it silence
            has_content = silence_ratio < 0.95

            return has_content

        except Exception:
            # If anything fails, assume it has content (safe fallback)
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

        logger.info("SAFE_AUDIO_LOG", extra=log_data)
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

        logger.info("SAFE_AUDIO_LOG", extra=log_data)

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
