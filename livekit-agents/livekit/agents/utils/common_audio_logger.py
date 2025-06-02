"""
Audio frame logging utilities for tracking audio frames through the LiveKit Agents pipeline.

This module provides functionality to track audio frames with unique IDs and timestamps
as they flow through different components of the audio processing pipeline.
"""

import time
import uuid
import weakref
from typing import Dict, Any, Optional

import numpy as np
from livekit import rtc
from livekit.agents.log import logger

# Stores {frame_object: {"id": uuid, "timestamps": {event_key: ns_timestamp}, "metadata": {}}}
# Using WeakKeyDictionary to prevent memory leaks from frame objects no longer in use.
frame_tracking_data: weakref.WeakKeyDictionary[rtc.AudioFrame, Dict[str, Any]] = (
    weakref.WeakKeyDictionary()
)


def calculate_dbfs_for_frame(frame_data) -> float:
    """
    Calculate dBFS (decibels relative to full scale) for an audio frame.

    Args:
        frame_data: Audio frame data, either memoryview or bytes

    Returns:
        float: dBFS value, or -inf if frame is empty/silent
    """
    try:
        if isinstance(frame_data, memoryview):
            audio_data = np.frombuffer(frame_data, dtype=np.int16)
        else:
            # Handle cases where frame_data might be bytes
            audio_data = np.frombuffer(frame_data.tobytes(), dtype=np.int16)
    except (TypeError, ValueError):
        return -float("inf")

    if len(audio_data) == 0:
        return -float("inf")

    audio_float = audio_data.astype(np.float64)
    rms = np.sqrt(np.mean(np.square(audio_float)))

    if rms == 0:
        return -float("inf")

    return 20 * np.log10(rms / 32767.0)


def record_audio_frame_timestamp(
    frame: rtc.AudioFrame,
    component_name: str,
    event_description: str,
    timestamp_ns: Optional[int] = None,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Records a timestamp for a given frame and event. Assigns an ID if the frame is new.

    Args:
        frame: The audio frame to track
        component_name: Name of the component processing the frame
        event_description: Description of the event/processing step
        timestamp_ns: Optional explicit timestamp (uses current time if None)
        additional_metadata: Optional metadata to store with the frame
    """
    ts = timestamp_ns if timestamp_ns is not None else time.time_ns()

    if frame not in frame_tracking_data:
        frame_id = uuid.uuid4()
        frame_tracking_data[frame] = {"id": frame_id, "timestamps": {}, "metadata": {}}

    entry = frame_tracking_data[frame]
    entry["timestamps"][f"{component_name}:{event_description}"] = ts

    if additional_metadata:
        entry["metadata"].update(additional_metadata)


def get_frame_id(frame: rtc.AudioFrame) -> Optional[uuid.UUID]:
    """
    Retrieves the UUID of a tracked frame.

    Args:
        frame: The audio frame to get ID for

    Returns:
        UUID of the frame if tracked, None otherwise
    """
    if frame in frame_tracking_data:
        return frame_tracking_data[frame]["id"]
    return None


def get_and_format_frame_journey(
    frame: rtc.AudioFrame,
    current_component: str,
    current_event: str,
    current_ts_ns: int,
    additional_info: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Retrieves all data for a frame and formats it for logging.
    This is intended to be called from the VAD plugin when final logging occurs.

    Args:
        frame: The audio frame to get journey for
        current_component: Current component name
        current_event: Current event description
        current_ts_ns: Current timestamp in nanoseconds
        additional_info: Additional information to include in the log

    Returns:
        Formatted log string or None if frame not tracked
    """
    if frame not in frame_tracking_data:
        return None

    entry = frame_tracking_data[frame]
    frame_id = entry["id"]

    # Sort timestamps by time for readability
    sorted_timestamps = sorted(entry["timestamps"].items(), key=lambda item: item[1])

    journey_str_parts = []
    first_ts = None
    for key, ts_val in sorted_timestamps:
        if first_ts is None:
            first_ts = ts_val
            journey_str_parts.append(f"{key} (Abs: {ts_val} ns, Rel: 0 ms)")
        else:
            relative_ms = (ts_val - first_ts) / 1_000_000
            journey_str_parts.append(
                f"{key} (Abs: {ts_val} ns, Rel: {relative_ms:.3f} ms)"
            )

    journey_str = " -> ".join(journey_str_parts)

    log_message = f"\nAudioFrame ID: {frame_id} - Journey: [ {journey_str} ]\n"

    if entry["metadata"]:
        log_message += f" - Metadata: {entry['metadata']}"

    if additional_info:
        log_message += f" - Additional Info: {additional_info}"

    return log_message
