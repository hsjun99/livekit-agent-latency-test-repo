# Audio Frame Logging Execution Plan

This document outlines the plan to log timestamps and other relevant information for `rtc.AudioFrame` objects as they flow through key components of the LiveKit Agents audio processing pipeline. The goal is to track the latency and behavior of audio frames, particularly focusing on their journey to and through the Silero VAD component.

## I. Core Requirements

1.  **Timestamping**: Log the `time.time_ns()` at each specified point of interest.
2.  **Frame Identification**: Assign a unique ID to each audio frame upon its first encounter (likely in `_ParticipantAudioInputStream`) to track it across different components. This might involve wrapping or extending the `rtc.AudioFrame` if it doesn't support custom attributes directly, or maintaining a separate tracking mechanism.
3.  **dBFS Calculation**: Implement and use the provided `calculate_dbfs` function within the Silero VAD plugin to log the dBFS of frames, but only if `dbfs > -float('inf')`.
4.  **Contextual Logging**: Logs should include the component name, method name, frame ID, timestamp, and any relevant stats (e.g., dBFS for VAD).
5.  **Output**: Logs should be directed to the standard logging mechanism (`logger`).
6.  **Performance**: Logging should be implemented efficiently to minimize performance impact on the audio processing pipeline.

## II. Frame ID Management Strategy

Since `rtc.AudioFrame` might not be directly extensible with a custom ID without modifying its core structure or `livekit-rtc` bindings, we will introduce a wrapper class or a parallel tracking mechanism.

**Option A: Wrapper Class (Preferred if `rtc.AudioFrame` needs to be passed around with extra data)**

```python
import time
import uuid
from livekit import rtc
import numpy as np # Ensure numpy is available where this is used

class TrackedAudioFrame:
    def __init__(self, frame: rtc.AudioFrame, source_component: str):
        self.id = uuid.uuid4()
        self.original_frame = frame
        self.timestamps: dict[str, int] = {} # component_method: timestamp_ns
        self.metadata: dict[str, any] = {}   # For additional data like dBFS
        self.log_event(f"{source_component}._init", time.time_ns())

    def log_event(self, event_name: str, timestamp_ns: int | None = None):
        timestamp_ns = timestamp_ns or time.time_ns()
        self.timestamps[event_name] = timestamp_ns
        # Potentially log directly here too, or aggregate and log later
        # print(f"Frame {self.id} event: {event_name} at {timestamp_ns}") # Placeholder

    @property
    def sample_rate(self) -> int:
        return self.original_frame.sample_rate

    @property
    def num_channels(self) -> int:
        return self.original_frame.num_channels

    @property
    def samples_per_channel(self) -> int:
        return self.original_frame.samples_per_channel

    @property
    def data(self) -> memoryview:
        return self.original_frame.data

    def calculate_dbfs(self) -> float:
        """Calculate dBFS (decibels relative to full scale) for an audio frame."""
        if isinstance(self.original_frame.data, memoryview):
            audio_data = np.frombuffer(self.original_frame.data, dtype=np.int16)
        else:
            # Fallback for older versions or different data types if necessary
            audio_data = np.frombuffer(self.original_frame.data.tobytes(), dtype=np.int16)

        if len(audio_data) == 0:
            return -float('inf')

        audio_float = audio_data.astype(np.float64)
        rms = np.sqrt(np.mean(np.square(audio_float)))

        if rms == 0:
            return -float('inf')

        return 20 * np.log10(rms / 32767.0)

    # Delegate other rtc.AudioFrame methods/properties if needed
```

If `TrackedAudioFrame` is used, all functions expecting `rtc.AudioFrame` will need to be updated to accept `TrackedAudioFrame` or to unwrap it. This could be invasive.

**Option B: External Tracking Dictionary (Less Invasive)**

Maintain a global or context-specific dictionary: `frame_tracking_data: weakref.WeakKeyDictionary[rtc.AudioFrame, dict]`
The dictionary would store `{ "id": uuid, "timestamps": {}, "metadata": {} }`.
`weakref.WeakKeyDictionary` is crucial here to prevent memory leaks if frames are discarded.

**Decision for Implementation:** We'll start with **Option B (External Tracking Dictionary with `weakref.WeakKeyDictionary`)** as it's less invasive. If direct modification of `rtc.AudioFrame` or a wrapper becomes necessary due to limitations, we can revisit. We will create a utility module for managing this tracking data.

## III. Points of Interest & Logging Actions

Audio frame events and timestamps will be accumulated. The actual consolidated logging will primarily occur within the Silero VAD plugin when specific conditions are met.

```python
# common_audio_logger.py (conceptual)
import time
import uuid
import weakref
import numpy as np
from livekit import rtc
from livekit.agents.log import logger # Assuming standard logger

# Stores {frame_object: {"id": uuid, "timestamps": {event_key: ns_timestamp}, "metadata": {}}}
# Using WeakKeyDictionary to prevent memory leaks from frame objects no longer in use.
frame_tracking_data = weakref.WeakKeyDictionary()

def calculate_dbfs_for_frame(frame_data: memoryview) -> float:
    # Ensure data is in a usable format (e.g., int16 numpy array)
    # This function might need to handle raw bytes if memoryview is not directly usable
    # or if frame.data is not always memoryview in all contexts.
    # For simplicity, assuming frame.data is a memoryview of int16 samples.
    try:
        audio_data = np.frombuffer(frame_data, dtype=np.int16)
    except TypeError: # Handle cases where frame_data might be bytes
        audio_data = np.frombuffer(frame_data.tobytes(), dtype=np.int16)

    if len(audio_data) == 0: return -float('inf')
    audio_float = audio_data.astype(np.float64)
    rms = np.sqrt(np.mean(np.square(audio_float)))
    if rms == 0: return -float('inf')
    return 20 * np.log10(rms / 32767.0)

def record_audio_frame_timestamp(frame: rtc.AudioFrame, component_name: str, event_description: str, additional_metadata: dict | None = None):
    """Records a timestamp for a given frame and event. Assigns an ID if the frame is new."""
    ts = time.time_ns()

    if frame not in frame_tracking_data:
        frame_id = uuid.uuid4()
        frame_tracking_data[frame] = {"id": frame_id, "timestamps": {}, "metadata": {}}

    entry = frame_tracking_data[frame]
    entry["timestamps"][f"{component_name}:{event_description}"] = ts

    if additional_metadata:
        entry["metadata"].update(additional_metadata)

def get_frame_id(frame: rtc.AudioFrame) -> uuid.UUID | None:
    """Retrieves the UUID of a tracked frame."""
    if frame in frame_tracking_data:
        return frame_tracking_data[frame]["id"]
    return None

def get_and_format_frame_journey(frame: rtc.AudioFrame, current_component: str, current_event: str, current_ts_ns: int, additional_info: dict | None = None) -> str | None:
    """
    Retrieves all data for a frame and formats it for logging.
    This is intended to be called from the VAD plugin when final logging occurs.
    """
    if frame not in frame_tracking_data:
        # If frame was never recorded (e.g., only VAD internal frames), log current info only
        # Or, decide if such frames should be tracked from their first appearance in VAD.
        # For now, assume primary tracking starts before VAD.
        # However, VAD creates its own event frames, which might need initial tracking if not passed from outside.
        # This logic might need refinement based on how VAD event frames are handled.

        # Let's ensure even VAD-originated event frames get an ID if they are to be logged.
        # The `record_audio_frame_timestamp` should be called first even in VAD for the current event.
        # So, this 'if' condition should ideally not be hit if `record_audio_frame_timestamp` was just called.
        return None

    entry = frame_tracking_data[frame]
    frame_id = entry["id"]

    # Add current event's timestamp to the journey, if not already added by record_audio_frame_timestamp
    # It's better if record_audio_frame_timestamp handles this to ensure atomicity with ID creation
    # entry["timestamps"][f"{current_component}:{current_event}"] = current_ts_ns

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
            journey_str_parts.append(f"{key} (Abs: {ts_val} ns, Rel: {relative_ms:.3f} ms)")

    journey_str = " -> ".join(journey_str_parts)

    log_message = f"AudioFrame ID: {frame_id} - Journey: [ {journey_str} ]"

    if entry["metadata"]:
        log_message += f" - Metadata: {entry['metadata']}"

    if additional_info: # Current VAD specific info
        log_message += f" - VAD Info: {additional_info}"

    return log_message

# The actual logger.info() will be called from the VAD plugin using the string from get_and_format_frame_journey
```

This utility will be imported and used in the target files.

---

### 1. `_ParticipantAudioInputStream` (`livekit/agents/voice/room_io/_input.py`)

- **Location**: Inside the `_forward_task` method.
- **Action**:
  - After frame is received from `stream`: Call `record_audio_frame_timestamp(frame, "_ParticipantAudioInputStream", "frame_received_from_rtc_stream")`.
  - Before `self._data_ch.send()`: Call `record_audio_frame_timestamp(frame, "_ParticipantAudioInputStream", "frame_sent_to_data_ch")`.

### 2. `AgentSession` (`livekit/agents/voice/agent_session.py`)

- **Location**: Inside `_forward_audio_task` method.
- **Action**:
  - After receiving frame from `audio_input`: Call `record_audio_frame_timestamp(frame, "AgentSession", "frame_received_from_audio_input")`.
  - Before `self._activity.push_audio(frame)`: Call `record_audio_frame_timestamp(frame, "AgentSession", "frame_pushed_to_activity")`.

### 3. `AgentActivity` (`livekit/agents/voice/agent_activity.py`)

- **Location**: Inside `push_audio` method.
- **Action**:
  - Upon method entry: Call `record_audio_frame_timestamp(frame, "AgentActivity", "frame_received")`.
  - Before `self._audio_recognition.push_audio(frame)`: Call `record_audio_frame_timestamp(frame, "AgentActivity", "frame_pushed_to_audio_recognition")`.

### 4. `AudioRecognition` (`livekit/agents/voice/audio_recognition.py`)

- **Location**: Inside `push_audio` method.
- **Action**:
  - Upon method entry: Call `record_audio_frame_timestamp(frame, "AudioRecognition", "frame_received")`.
  - If `_stt_ch` is active, before `self._stt_ch.send_nowait(frame)`: Call `record_audio_frame_timestamp(frame, "AudioRecognition", "frame_sent_to_stt_ch")`.
  - If `_vad_ch` is active, before `self._vad_ch.send_nowait(frame)`: Call `record_audio_frame_timestamp(frame, "AudioRecognition", "frame_sent_to_vad_ch")`.

### 5. Silero VAD Stream (`livekit-plugins/livekit-plugins-silero/livekit/plugins/silero/vad.py`)

This section details how `VADStream._main_task` will handle recording timestamps and performing the final consolidated logging for individual frames _as they are received by the VAD plugin, before any internal buffering or combining for inference_.

- **Imports in `vad.py`**:

  - Ensure `numpy` is imported.
  - From `common_audio_logger` (adjust path as needed, e.g., `livekit.agents.utils.common_audio_logger`), import `record_audio_frame_timestamp`, `calculate_dbfs_for_frame`, `get_and_format_frame_journey`.
  - Import `logger` from `livekit.agents.log` (or use an existing VAD-specific logger if preferred).

- **Logging Logic in `VADStream._main_task`**:

  The primary logging action will occur immediately after a frame is received from the input channel.

  ```python
  # Inside VADStream._main_task, within the loop processing input_ch:
  # async for input_frame in self._input_ch:
  #     ...

  async for input_frame in self._input_ch:  # This is the frame from AudioRecognition
      current_ts_ns = time.time_ns()

      # 1. Record timestamp for frame arrival at VAD.
      # This also ensures the frame gets an ID if it's new to the tracking system.
      record_audio_frame_timestamp(
          frame=input_frame,
          component_name="SileroVADStream",
          event_description="frame_received_from_input_ch",
          timestamp_ns=current_ts_ns
      )

      # 2. Calculate dBFS for this specific input_frame.
      dbfs_current_frame = calculate_dbfs_for_frame(input_frame.data)

      # 3. Update metadata for this input_frame with its dBFS.
      # This ensures dBFS is part of its tracked metadata before final log formatting.
      # Note: This call uses the *same* input_frame object, so it updates its existing entry.
      record_audio_frame_timestamp(
          frame=input_frame,
          component_name="SileroVADStream",
          event_description="dbfs_calculated_for_input_frame",
          # timestamp_ns is not strictly needed here if we consider this part of the "frame_received" event's processing instant,
          # but using current_ts_ns is fine. The main purpose is to add metadata.
          timestamp_ns=current_ts_ns, # Or a new time.time_ns() if measuring calc time is desired for this specific step
          additional_metadata={"dbfs_input_frame": round(dbfs_current_frame, 2) if dbfs_current_frame > -float('inf') else -float('inf')}
      )

      # 4. Conditional Final Logging for this input_frame's journey.
      if dbfs_current_frame > -float('inf'):
          # The additional_info here will be merged with any metadata already recorded for the frame by get_and_format_frame_journey.
          # Since dbfs_input_frame was just recorded into metadata, it will be available.
          log_entry = get_and_format_frame_journey(
              frame=input_frame,
              final_event_timestamp_ns=current_ts_ns, # Use current time as the reference for this log event
              additional_info={"log_trigger_reason": "input_frame_significant_dbfs_at_vad_entry"}
          )
          if log_entry:
              logger.info(log_entry)

      # --- Original VAD processing continues below ---
      # The input_frame is then typically added to a buffer (e.g., self._input_frames.append(input_frame))
      # and processed further (e.g., utils.combine_frames, inference).
      # The detailed journey logging for those combined/processed frames is no longer the primary focus.
      # Basic operational logging for VAD (like inference probability on a combined window)
      # can remain if necessary but will not use get_and_format_frame_journey for those internal/combined frames.

      # Example of original processing continuing (simplified):
      # self._input_frames.append(input_frame)
      # if len(self._input_frames_bytes) >= self._model.window_size_bytes:
      #     # ... logic to combine frames ...
      #     # combined_frame = utils.combine_frames(self._input_frames, self._model.window_size_bytes)
      #     # ... VAD inference on combined_frame ...
      #     # p = await self._loop.run_in_executor(...)
      #     # logger.debug(f"VAD inference probability: {p} for combined window") # Example of simplified/existing log
      #     # ... rest of VAD logic for emitting VADEvents ...
  ```

- **Considerations**:
  - This change focuses the detailed journey logging on individual `rtc.AudioFrame` objects as they enter the VAD, right before they are internally buffered or combined.
  - Any existing simpler logging within the VAD for its operational steps (like inference probability on combined windows, or VAD events being triggered based on inference results) can remain but should not use `get_and_format_frame_journey` for those combined/event frames, to keep the focus on the incoming frame's journey.
  - The `final_event_timestamp_ns` passed to `get_and_format_frame_journey` is the timestamp of the event that triggers the log, which in this case is the frame's arrival at VAD and its dBFS calculation.

## IV. Implementation Steps

1.  **Create `common_audio_logger.py`**:
    - Path: `livekit/agents/utils/common_audio_logger.py`
    - Define `frame_tracking_data` (`weakref.WeakKeyDictionary`).
    - Define `calculate_dbfs_for_frame`.
    - Define `record_audio_frame_timestamp`.
    - Define `get_frame_id`.
    - Define `get_and_format_frame_journey`.
    - Ensure `numpy` is available and imported.
2.  **Modify `_ParticipantAudioInputStream`** (`livekit/agents/voice/room_io/_input.py`):
    - Import `record_audio_frame_timestamp` from `livekit.agents.utils.common_audio_logger`.
    - In `_ParticipantInputStream._forward_task` (the base class method):
      - After frame is received from `stream`: Call `record_audio_frame_timestamp(event.frame, "_ParticipantInputStream", "frame_received_from_rtc_stream")`.
      - Before `self._data_ch.send()`: Call `record_audio_frame_timestamp(event.frame, "_ParticipantInputStream", "frame_sent_to_data_ch")`.
    - In `_ParticipantAudioInputStream._forward_task` (the overridden method for pre-connect audio):
      - Before sending a resampled pre-connect frame to `_data_ch`: Call `record_audio_frame_timestamp(frame, "_ParticipantAudioInputStream", "pre_connect_frame_processed_by_resampler")` and `record_audio_frame_timestamp(frame, "_ParticipantAudioInputStream", "pre_connect_frame_about_to_send_to_data_ch")`.
3.  **Modify `AgentSession`** (`livekit/agents/voice/agent_session.py`):
    - Import `record_audio_frame_timestamp` from `livekit.agents.utils.common_audio_logger`.
    - In `_forward_audio_task`:
      - After receiving frame from `audio_input`: Call `record_audio_frame_timestamp(frame, "AgentSession", "frame_received_from_audio_input")`.
      - Before `self._activity.push_audio(frame)`: Call `record_audio_frame_timestamp(frame, "AgentSession", "frame_pushed_to_activity")`.
4.  **Modify `AgentActivity`** (`livekit/agents/voice/agent_activity.py`):
    - Import `record_audio_frame_timestamp` from `livekit.agents.utils.common_audio_logger`.
    - In `push_audio` method:
      - Upon method entry: Call `record_audio_frame_timestamp(frame, "AgentActivity", "frame_received")`.
      - Before `self._audio_recognition.push_audio(frame)`: Call `record_audio_frame_timestamp(frame, "AgentActivity", "frame_pushed_to_audio_recognition")`.
5.  **Modify `AudioRecognition`** (`livekit/agents/voice/audio_recognition.py`):
    - Import `record_audio_frame_timestamp` from `livekit.agents.utils.common_audio_logger`.
    - In `push_audio` method:
      - Upon method entry: Call `record_audio_frame_timestamp(frame, "AudioRecognition", "frame_received")`.
      - If `_stt_ch` is active, before `self._stt_ch.send_nowait(frame)`: Call `record_audio_frame_timestamp(frame, "AudioRecognition", "frame_sent_to_stt_ch")`.
      - If `_vad_ch` is active, before `self._vad_ch.send_nowait(frame)`: Call `record_audio_frame_timestamp(frame, "AudioRecognition", "frame_sent_to_vad_ch")`.
6.  **Modify Silero VAD Plugin** (`livekit-plugins/livekit-plugins-silero/livekit/plugins/silero/vad.py`):
    - Import necessary functions from `livekit.agents.utils.common_audio_logger` (`record_audio_frame_timestamp`, `calculate_dbfs_for_frame`, `get_and_format_frame_journey`).
    - Import `logger` from `livekit.agents.log`.
    - Add `numpy` import if missing.
    - Modify `VADStream._main_task` as specified in section "### 5. Silero VAD Stream" above, focusing on logging individual frames upon reception from `_input_ch` and their dBFS.
7.  **Testing and Refinement**:
    - Run a simple agent scenario.
    - Inspect logs to verify timestamps, frame ID tracking, dBFS calculations, and correct event ordering.
    - Profile for any significant performance degradation.

## V. Considerations for Frame Identity in VAD

- The `VADStream` internally buffers and processes audio in windows (`self._model.window_size_samples`). The `input_frame` variable inside `_main_task`'s `while True` loop (after `utils.combine_frames`) represents this processing window, not necessarily a single original `rtc.AudioFrame` if multiple small frames were combined.
- Logging based on this combined `input_frame` for "pre_inference_window" and "post_inference_window" events makes sense for understanding what data window the VAD model saw.
- The `frames` field in `VADEvent` (for `START_OF_SPEECH`, `END_OF_SPEECH`) contains actual segments of audio that contributed to the event. Logging these with their original IDs (if traceable through `combine_frames` or by logging data rather than frame objects for these specific events) is valuable.
- The `common_audio_logger` using `weakref.WeakKeyDictionary` on `rtc.AudioFrame` objects assumes that the _identity_ of frame objects passed around is somewhat preserved or that new frame objects created from data segments are distinct. If `utils.combine_frames` or other operations create new `rtc.AudioFrame` objects from combined data, the original IDs might be lost unless the tracking mechanism is more sophisticated (e.g., content hashing, which is too complex here, or ensuring the `combine_frames` utility can carry over metadata if a wrapper is used).
- For Silero VAD, the most important part is to log the data that _actually goes into the model_ and the _data segments that form speech events_. The `input_frame` (combined) and `ev.frames` in `VADEvent` are the best proxies for these.

## VI. Summary of Changes (File by File)

1.  **NEW**: `livekit/agents/utils/common_audio_logger.py`
    - Contains the logging utilities: `frame_tracking_data`, `calculate_dbfs_for_frame`, `record_audio_frame_timestamp`, `get_frame_id`, `get_and_format_frame_journey`.
2.  `livekit/agents/voice/room_io/_input.py`
    - Add import for `common_audio_logger`.
    - Modify `_ParticipantInputStream._forward_task` and `_ParticipantAudioInputStream._forward_task` for pre-connect audio.
3.  `livekit/agents/voice/agent_session.py`
    - Add import for `common_audio_logger`.
    - Modify `AgentSession._forward_audio_task`.
4.  `livekit/agents/voice/agent_activity.py`
    - Add import for `common_audio_logger`.
    - Modify `AgentActivity.push_audio`.
5.  `livekit/agents/voice/audio_recognition.py`
    - Add import for `common_audio_logger`.
    - Modify `AudioRecognition.push_audio`.
6.  `livekit-plugins/livekit-plugins-silero/livekit/plugins/silero/vad.py`
    - Add import for `common_audio_logger` and `livekit.agents.log.logger`.
    - Add `numpy` import if missing.
    - Modify `VADStream._main_task` to log individual incoming frames with their dBFS and journey, as detailed in the updated plan. Deprioritize detailed journey logging for combined/event frames.

This plan provides a clear path to achieving the desired logging. The key challenge will be managing frame identity correctly, especially within the VAD plugin's buffering and windowing mechanisms.
The current approach of focusing on individual frames as they arrive at the VAD simplifies some of these identity concerns for the primary log output.
