# LiveKit Agents Simple but Thorough Audio Logging Plan

## Overview

This document provides a comprehensive but simplified logging plan for audio analysis in LiveKit agents. The goal is to capture every audio processing operation with precise timing, using only standard Python libraries.

## 1. Simple Audio Analysis System

### 1.1 Basic Audio Metrics

**File**: Create `/livekit/agents/utils/audio_logging.py`

```python
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
    dbfs: float              # dBFS (decibels relative to full scale)
    rms_energy: float        # Root Mean Square energy
    peak_amplitude: float    # Peak amplitude (0.0-1.0)
    zero_crossing_rate: float # Zero crossing rate
    spectral_centroid: float  # Simple frequency content measure
    has_audio_content: bool   # Whether frame contains meaningful audio

def calculate_audio_metrics(frame: rtc.AudioFrame) -> SimpleAudioMetrics:
    """Calculate comprehensive audio metrics using only numpy."""

    # Convert frame data to numpy array
    if isinstance(frame.data, memoryview):
        audio_data = np.frombuffer(frame.data, dtype=np.int16)
    else:
        audio_data = np.frombuffer(frame.data.tobytes(), dtype=np.int16)

    if len(audio_data) == 0:
        return SimpleAudioMetrics(
            dbfs=-float('inf'), rms_energy=0.0, peak_amplitude=0.0,
            zero_crossing_rate=0.0, spectral_centroid=0.0, has_audio_content=False
        )

    # Convert to float64 for precision
    audio_float = audio_data.astype(np.float64)

    # 1. dBFS calculation
    rms = np.sqrt(np.mean(np.square(audio_float)))
    if rms == 0:
        dbfs = -float('inf')
    else:
        dbfs = 20 * math.log10(rms / 32767.0)

    # 2. RMS Energy (normalized)
    rms_energy = rms / 32767.0

    # 3. Peak amplitude
    peak_amplitude = np.max(np.abs(audio_float)) / 32767.0

    # 4. Zero Crossing Rate
    zero_crossings = np.where(np.diff(np.signbit(audio_float)))[0]
    zero_crossing_rate = len(zero_crossings) / len(audio_float) if len(audio_float) > 1 else 0.0

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
        dbfs > -50.0 and                    # Sufficient volume
        rms_energy > 0.001 and             # Minimum energy
        zero_crossing_rate > 0.01           # Some frequency content
    )

    return SimpleAudioMetrics(
        dbfs=round(dbfs, 2),
        rms_energy=round(rms_energy, 4),
        peak_amplitude=round(peak_amplitude, 4),
        zero_crossing_rate=round(zero_crossing_rate, 4),
        spectral_centroid=round(spectral_centroid, 1),
        has_audio_content=has_audio_content
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
        "timestamp_ms": current_time // 1_000_000
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
    return (
        metrics.has_audio_content or
        location in ["webrtc_input", "audio_recognition_input", "vad_inference_complete"]
    )

def log_audio_frame(frame: rtc.AudioFrame, location: str, frame_id: str = None,
                   extra_data: dict = None) -> Optional[str]:
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
        "has_audio_content": metrics.has_audio_content
    }

    if extra_data:
        log_data.update(extra_data)

    logger.info("AUDIO_FLOW", extra=log_data)
    return frame_id

def log_processing_step(frame_id: str, location: str, operation: str,
                       start_time_ns: int, end_time_ns: int, extra_data: dict = None):
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
        **timing
    }

    if extra_data:
        log_data.update(extra_data)

    logger.info("PROCESSING_STEP", extra=log_data)

def log_memory_operation(frame_id: str, location: str, operation: str,
                        size_bytes: int, duration_ns: int):
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
        **timing
    }

    logger.info("MEMORY_OPERATION", extra=log_data)

def log_channel_operation(frame_id: str, location: str, operation: str,
                         duration_ns: int, queue_size: int = None):
    """Log async channel operations."""
    timing = get_precise_timing()

    log_data = {
        "frame_id": frame_id,
        "location": location,
        "operation": operation,
        "duration_ns": duration_ns,
        "duration_ms": round(duration_ns / 1_000_000, 3),
        **timing
    }

    if queue_size is not None:
        log_data["queue_size"] = queue_size

    logger.info("CHANNEL_OPERATION", extra=log_data)
```

## 2. Detailed Logging Points

### 2.1 WebRTC Audio Reception

**File**: `/livekit/agents/voice/room_io/_input.py`

**Function**: `_ParticipantAudioInputStream._forward_task()`

```python
async def _forward_task(self, old_task, stream, publication, participant):
    if old_task:
        await aio.cancel_and_wait(old_task)

    extra = {
        "participant": participant.identity,
        "source": rtc.TrackSource.Name(publication.source),
    }
    logger.debug("start reading stream", extra=extra)

    async for event in stream:
        if not self._attached:
            continue

        # LOG: Raw WebRTC frame reception
        reception_start_ns = time.time_ns()
        frame_id = log_audio_frame(
            cast(T, event.frame),
            location="webrtc_input",
            extra_data={
                "participant": participant.identity,
                "source": rtc.TrackSource.Name(publication.source),
                "stream_attached": self._attached
            }
        )

        # LOG: Frame copy operation
        copy_start_ns = time.time_ns()
        frame_copy = cast(T, event.frame)
        copy_end_ns = time.time_ns()

        log_memory_operation(
            frame_id=frame_id,
            location="webrtc_frame_copy",
            operation="frame_copy",
            size_bytes=len(event.frame.data),
            duration_ns=copy_end_ns - copy_start_ns
        )

        # LOG: Channel send operation
        send_start_ns = time.time_ns()
        await self._data_ch.send(frame_copy)
        send_end_ns = time.time_ns()

        log_channel_operation(
            frame_id=frame_id,
            location="webrtc_to_room_io",
            operation="async_channel_send",
            duration_ns=send_end_ns - send_start_ns,
            queue_size=getattr(self._data_ch, 'qsize', lambda: None)()
        )

        # LOG: Complete packet processing
        reception_end_ns = time.time_ns()
        log_processing_step(
            frame_id=frame_id,
            location="webrtc_packet_complete",
            operation="complete_packet_processing",
            start_time_ns=reception_start_ns,
            end_time_ns=reception_end_ns,
            extra_data={
                "copy_time_ns": copy_end_ns - copy_start_ns,
                "send_time_ns": send_end_ns - send_start_ns
            }
        )

    logger.debug("stream closed", extra=extra)
```

### 2.2 Pre-Connect Audio Handling

**File**: `/livekit/agents/voice/room_io/_input.py`

**Function**: `_ParticipantAudioInputStream._forward_task()` (pre-connect section)

```python
# Handle pre-connect audio
if (self._pre_connect_audio_handler
    and publication.track
    and AudioTrackFeature.TF_PRECONNECT_BUFFER in publication.audio_features):

    logging_extra = {
        "track_id": publication.track.sid,
        "participant": participant.identity,
    }

    try:
        preconnect_start_ns = time.time_ns()

        duration: float = 0
        frames = await self._pre_connect_audio_handler.wait_for_data(publication.track.sid)

        preconnect_receive_ns = time.time_ns()

        # LOG: Pre-connect buffer received
        logger.info("PRECONNECT_BUFFER_RECEIVED", extra={
            "track_id": publication.track.sid,
            "frames_count": len(frames),
            "wait_time_ms": round((preconnect_receive_ns - preconnect_start_ns) / 1_000_000, 3),
            **logging_extra
        })

        for frame in self._resample_frames(frames):
            if self._attached:
                # LOG: Pre-connect frame processing
                frame_id = log_audio_frame(
                    frame,
                    location="preconnect_audio",
                    extra_data={
                        "track_id": publication.track.sid,
                        "buffer_duration_ms": round(duration * 1000, 2)
                    }
                )

                await self._data_ch.send(frame)
                duration += frame.duration

        if frames:
            logger.debug("pre-connect audio buffer pushed",
                        extra={"duration": duration, **logging_extra})

    except asyncio.TimeoutError:
        logger.warning("timeout waiting for pre-connect audio buffer", extra=logging_extra)
    except Exception as e:
        logger.error("error reading pre-connect audio buffer", extra=logging_extra, exc_info=e)
```

### 2.3 Audio Resampling

**File**: `/livekit/agents/voice/room_io/_input.py`

**Function**: `_ParticipantAudioInputStream._resample_frames()`

```python
def _resample_frames(self, frames: Iterable[rtc.AudioFrame]) -> Iterable[rtc.AudioFrame]:
    resampler: rtc.AudioResampler | None = None

    for frame in frames:
        resample_start_ns = time.time_ns()

        # LOG: Input frame to resampling
        frame_id = log_audio_frame(
            frame,
            location="resampling_input",
            extra_data={
                "input_rate": frame.sample_rate,
                "target_rate": self._sample_rate,
                "resampling_needed": frame.sample_rate != self._sample_rate
            }
        )

        if (not resampler and self._sample_rate is not None
            and frame.sample_rate != self._sample_rate):

            # LOG: Resampler creation
            resampler_create_start = time.time_ns()
            resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate,
                output_rate=self._sample_rate
            )
            resampler_create_end = time.time_ns()

            log_processing_step(
                frame_id=frame_id,
                location="resampler_creation",
                operation="create_resampler",
                start_time_ns=resampler_create_start,
                end_time_ns=resampler_create_end,
                extra_data={
                    "input_rate": frame.sample_rate,
                    "output_rate": self._sample_rate
                }
            )

        if resampler:
            # LOG: Resampling operation
            resample_op_start = time.time_ns()
            resampled_frames_list = list(resampler.push(frame))
            resample_op_end = time.time_ns()

            log_processing_step(
                frame_id=frame_id,
                location="resampling_operation",
                operation="audio_resampling",
                start_time_ns=resample_op_start,
                end_time_ns=resample_op_end,
                extra_data={
                    "input_samples": frame.samples_per_channel,
                    "output_frames": len(resampled_frames_list),
                    "total_output_samples": sum(f.samples_per_channel for f in resampled_frames_list)
                }
            )

            # LOG: Each resampled output frame
            for i, resampled_frame in enumerate(resampled_frames_list):
                output_frame_id = log_audio_frame(
                    resampled_frame,
                    location="resampling_output",
                    frame_id=f"{frame_id}_out_{i}",
                    extra_data={
                        "original_frame_id": frame_id,
                        "resampled_index": i
                    }
                )
                yield resampled_frame
        else:
            # No resampling needed
            passthrough_end_ns = time.time_ns()

            log_processing_step(
                frame_id=frame_id,
                location="resampling_passthrough",
                operation="passthrough_no_resampling",
                start_time_ns=resample_start_ns,
                end_time_ns=passthrough_end_ns,
                extra_data={"reason": "matching_sample_rates"}
            )

            yield frame
```

### 2.4 Audio Recognition Distribution

**File**: `/livekit/agents/voice/audio_recognition.py`

**Function**: `AudioRecognition.push_audio()`

```python
def push_audio(self, frame: rtc.AudioFrame) -> None:
    distribution_start_ns = time.time_ns()

    self._sample_rate = frame.sample_rate

    # LOG: Critical distribution point
    frame_id = log_audio_frame(
        frame,
        location="audio_recognition_input",
        extra_data={
            "distribution_target": "stt_and_vad",
            "stt_enabled": self._stt_ch is not None,
            "vad_enabled": self._vad_ch is not None
        }
    )

    # STT path
    if self._stt_ch is not None:
        stt_send_start = time.time_ns()
        self._stt_ch.send_nowait(frame)
        stt_send_end = time.time_ns()

        log_channel_operation(
            frame_id=frame_id,
            location="stt_channel_send",
            operation="send_to_stt",
            duration_ns=stt_send_end - stt_send_start,
            queue_size=getattr(self._stt_ch, 'qsize', lambda: None)()
        )

    # VAD path
    if self._vad_ch is not None:
        vad_send_start = time.time_ns()
        self._vad_ch.send_nowait(frame)
        vad_send_end = time.time_ns()

        log_channel_operation(
            frame_id=frame_id,
            location="vad_channel_send",
            operation="send_to_vad",
            duration_ns=vad_send_end - vad_send_start,
            queue_size=getattr(self._vad_ch, 'qsize', lambda: None)()
        )

    distribution_end_ns = time.time_ns()

    # LOG: Complete distribution
    log_processing_step(
        frame_id=frame_id,
        location="audio_recognition_distribution_complete",
        operation="distribute_to_stt_and_vad",
        start_time_ns=distribution_start_ns,
        end_time_ns=distribution_end_ns
    )
```

### 2.5 VAD Processing

**File**: `/livekit/plugins/silero/vad.py`

**Function**: `VADStream._main_task()`

```python
async def _main_task(self):
    inference_f32_data = np.empty(self._model.window_size_samples, dtype=np.float32)
    # ... other initialization ...

    async for input_frame in self._input_ch:
        frame_receive_ns = time.time_ns()

        if not isinstance(input_frame, rtc.AudioFrame):
            continue

        # LOG: VAD frame dequeue
        frame_id = log_audio_frame(
            input_frame,
            location="vad_frame_dequeue",
            extra_data={
                "queue_depth": getattr(self._input_ch, 'qsize', lambda: None)(),
                "vad_type": "silero"
            }
        )

        # ... existing frame accumulation logic ...

        while True:
            inference_cycle_start = time.time_ns()

            available_inference_samples = sum([frame.samples_per_channel for frame in inference_frames])
            if available_inference_samples < self._model.window_size_samples:
                break

            # LOG: Frame combination for VAD window
            combine_start = time.time_ns()
            input_frame = utils.combine_frames(input_frames)
            inference_frame = utils.combine_frames(inference_frames)
            combine_end = time.time_ns()

            log_processing_step(
                frame_id=frame_id,
                location="vad_frame_combination",
                operation="combine_frames_for_window",
                start_time_ns=combine_start,
                end_time_ns=combine_end,
                extra_data={
                    "frames_combined": len(inference_frames),
                    "window_size_samples": self._model.window_size_samples,
                    "total_samples": available_inference_samples
                }
            )

            # LOG: Data format conversion
            conversion_start = time.time_ns()
            np.divide(
                inference_frame.data[: self._model.window_size_samples],
                np.iinfo(np.int16).max,
                out=inference_f32_data,
                dtype=np.float32,
            )
            conversion_end = time.time_ns()

            log_processing_step(
                frame_id=frame_id,
                location="vad_data_conversion",
                operation="int16_to_float32",
                start_time_ns=conversion_start,
                end_time_ns=conversion_end,
                extra_data={
                    "samples_converted": self._model.window_size_samples,
                    "input_dtype": "int16",
                    "output_dtype": "float32"
                }
            )

            # CRITICAL: VAD Inference
            inference_start = time.time_ns()

            # Calculate input audio metrics for context
            input_metrics = calculate_audio_metrics(inference_frame)

            logger.info("VAD_INFERENCE_START", extra={
                "frame_id": frame_id,
                "location": "vad_inference_start",
                "model_type": "silero_onnx",
                "window_size": self._model.window_size_samples,
                "input_dbfs": input_metrics.dbfs,
                "input_has_content": input_metrics.has_audio_content,
                "timestamp_ns": inference_start
            })

            p = await self._loop.run_in_executor(
                self._executor, self._model, inference_f32_data
            )

            inference_end = time.time_ns()
            inference_duration = (inference_end - inference_start) / 1_000_000

            # Exponential filtering
            filter_start = time.time_ns()
            p_filtered = self._exp_filter.apply(exp=1.0, sample=p)
            filter_end = time.time_ns()

            # LOG: Complete VAD inference results
            logger.info("VAD_INFERENCE_COMPLETE", extra={
                "frame_id": frame_id,
                "location": "vad_inference_complete",
                "raw_probability": round(p, 6),
                "filtered_probability": round(p_filtered, 6),
                "inference_latency_ms": round(inference_duration, 3),
                "filter_time_ns": filter_end - filter_start,
                "exceeds_threshold": p_filtered >= self._opts.activation_threshold,
                "activation_threshold": self._opts.activation_threshold,
                "input_audio_metrics": input_metrics.__dict__,
                "timestamp_ns": inference_end
            })

            # Track slow inference
            if inference_duration > 50:  # >50ms is concerning
                logger.warning("SLOW_VAD_INFERENCE", extra={
                    "frame_id": frame_id,
                    "inference_latency_ms": inference_duration,
                    "expected_max_ms": 50,
                    "timestamp_ns": inference_end
                })

            # ... continue with existing speech detection logic ...
```

### 2.6 General STT Processing (Audio Recognition Level)

**File**: `/livekit/agents/voice/audio_recognition.py`

**Function**: `AudioRecognition._stt_task()`

```python
@log_exceptions(logger=logger)
async def _stt_task(self):
    stt_start_ns = time.time_ns()

    # LOG: STT stream initialization
    logger.info("STT_STREAM_START", extra={
        "stt_provider": type(self._stt).__name__,
        "sample_rate": self._sample_rate,
        "timestamp_ns": stt_start_ns
    })

    async for frame in self._stt_ch:
        frame_receive_ns = time.time_ns()

        # LOG: STT frame processing
        frame_id = log_audio_frame(
            frame,
            location="stt_stream_input",
            extra_data={
                "stt_provider": type(self._stt).__name__,
                "queue_depth": getattr(self._stt_ch, 'qsize', lambda: None)()
            }
        )

        # Forward to STT provider
        stt_forward_start = time.time_ns()
        # Note: This goes to the specific STT implementation
        # We're not modifying the plugin directly, just logging at this level
        stt_forward_end = time.time_ns()

        log_processing_step(
            frame_id=frame_id,
            location="stt_provider_forward",
            operation="forward_to_stt_provider",
            start_time_ns=stt_forward_start,
            end_time_ns=stt_forward_end,
            extra_data={
                "provider": type(self._stt).__name__
            }
        )
```

### 2.7 AudioByteStream Processing

**File**: `/livekit/agents/utils/audio.py`

**Function**: `AudioByteStream.push()`

```python
def push(self, frame: rtc.AudioFrame) -> Iterator[rtc.AudioFrame]:
    push_start_ns = time.time_ns()

    # LOG: AudioByteStream input
    frame_id = log_audio_frame(
        frame,
        location="audiobytestream_input",
        extra_data={
            "target_chunk_size_ms": round((self._bytes_per_frame / self._bytes_per_sample) / self._sample_rate * 1000, 2),
            "buffer_fill_bytes": len(self._buf),
            "frame_size_bytes": len(frame.data)
        }
    )

    # Memory operation: Buffer append
    append_start = time.time_ns()
    self._buf.extend(frame.data)
    append_end = time.time_ns()

    log_memory_operation(
        frame_id=frame_id,
        location="audiobytestream_buffer_append",
        operation="buffer_extend",
        size_bytes=len(frame.data),
        duration_ns=append_end - append_start
    )

    chunk_count = 0
    while len(self._buf) >= self._bytes_per_frame:
        chunk_start = time.time_ns()

        # Memory operation: Chunk extraction
        extract_start = time.time_ns()
        chunk_data = self._buf[:self._bytes_per_frame]
        self._buf = self._buf[self._bytes_per_frame:]
        extract_end = time.time_ns()

        # Create output frame
        frame_create_start = time.time_ns()
        chunk_frame = rtc.AudioFrame(
            data=chunk_data,
            sample_rate=self._sample_rate,
            num_channels=self._num_channels,
            samples_per_channel=self._bytes_per_frame // self._bytes_per_sample,
        )
        frame_create_end = time.time_ns()

        chunk_end = time.time_ns()
        chunk_count += 1

        # LOG: Chunk output
        chunk_frame_id = log_audio_frame(
            chunk_frame,
            location="audiobytestream_chunk_output",
            frame_id=f"{frame_id}_chunk_{chunk_count}",
            extra_data={
                "original_frame_id": frame_id,
                "chunk_number": chunk_count,
                "buffer_remaining_bytes": len(self._buf)
            }
        )

        log_processing_step(
            frame_id=chunk_frame_id,
            location="audiobytestream_chunk_creation",
            operation="create_audio_chunk",
            start_time_ns=chunk_start,
            end_time_ns=chunk_end,
            extra_data={
                "extract_time_ns": extract_end - extract_start,
                "frame_create_time_ns": frame_create_end - frame_create_start,
                "chunk_size_bytes": len(chunk_data)
            }
        )

        yield chunk_frame

    push_end_ns = time.time_ns()

    # LOG: Complete push operation
    log_processing_step(
        frame_id=frame_id,
        location="audiobytestream_push_complete",
        operation="complete_audiobytestream_push",
        start_time_ns=push_start_ns,
        end_time_ns=push_end_ns,
        extra_data={
            "chunks_produced": chunk_count,
            "buffer_remaining": len(self._buf)
        }
    )
```

## 3. Configuration and Usage

### 3.1 Simple Configuration

```python
SIMPLE_AUDIO_LOGGING_CONFIG = {
    "enabled": True,
    "min_dbfs_threshold": -50.0,  # Log audio above this level
    "locations": {
        "webrtc_input": True,
        "audio_recognition_input": True,
        "vad_inference_complete": True,
        "stt_stream_input": True,
        "audiobytestream_chunk_output": True
    },
    "log_memory_operations": True,
    "log_channel_operations": True,
    "log_processing_steps": True
}
```

### 3.2 Analysis Functions

```python
def analyze_hello_latency(logs: list) -> dict:
    """Analyze latency for 'hello' audio through pipeline."""

    # Filter for audio frames with content
    audio_logs = [log for log in logs
                 if log.get("extra", {}).get("has_audio_content", False)]

    # Group by frame_id
    frame_groups = {}
    for log in audio_logs:
        frame_id = log.get("extra", {}).get("frame_id")
        if frame_id:
            if frame_id not in frame_groups:
                frame_groups[frame_id] = []
            frame_groups[frame_id].append(log)

    # Calculate latencies for each frame
    latency_analysis = {}
    for frame_id, frame_logs in frame_groups.items():
        # Sort by timestamp
        sorted_logs = sorted(frame_logs, key=lambda x: x["extra"]["timestamp_ns"])

        if len(sorted_logs) > 1:
            first_timestamp = sorted_logs[0]["extra"]["timestamp_ns"]
            latencies = {}

            for log in sorted_logs:
                location = log["extra"]["location"]
                latency_ms = (log["extra"]["timestamp_ns"] - first_timestamp) / 1_000_000
                latencies[location] = round(latency_ms, 3)

            latency_analysis[frame_id] = latencies

    return latency_analysis

def find_bottlenecks(logs: list) -> dict:
    """Find processing bottlenecks from logs."""

    processing_times = {}

    for log in logs:
        if log.get("message") == "PROCESSING_STEP":
            location = log.get("extra", {}).get("location")
            processing_time = log.get("extra", {}).get("processing_time_ms", 0)

            if location:
                if location not in processing_times:
                    processing_times[location] = []
                processing_times[location].append(processing_time)

    # Calculate statistics
    bottlenecks = {}
    for location, times in processing_times.items():
        if times:
            avg_time = sum(times) / len(times)
            max_time = max(times)

            if avg_time > 1.0:  # >1ms average
                bottlenecks[location] = {
                    "avg_time_ms": round(avg_time, 3),
                    "max_time_ms": round(max_time, 3),
                    "sample_count": len(times)
                }

    return bottlenecks
```

## 4. Expected Output for "Hello"

When you say "hello", you'll see logs like:

```json
{"message": "AUDIO_FLOW", "extra": {"location": "webrtc_input", "frame_id": "frame_00000123", "dbfs": -22.5, "has_audio_content": true, "timestamp_ns": 1703123456789000}}

{"message": "PROCESSING_STEP", "extra": {"frame_id": "frame_00000123", "location": "webrtc_packet_complete", "processing_time_ms": 0.125}}

{"message": "AUDIO_FLOW", "extra": {"location": "audio_recognition_input", "frame_id": "frame_00000123", "distribution_target": "stt_and_vad"}}

{"message": "VAD_INFERENCE_COMPLETE", "extra": {"frame_id": "frame_00000123", "filtered_probability": 0.1234, "inference_latency_ms": 5.2}}
```

This simplified plan gives you thorough coverage of the audio pipeline without complex dependencies, focusing on the essential timing and processing information you need to analyze your 150-200ms latency.
