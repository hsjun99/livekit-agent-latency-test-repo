# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

import numpy as np
import onnxruntime  # type: ignore

from livekit import agents, rtc
from livekit.agents import utils
from livekit.agents.types import (
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from livekit.agents.log import logger
from livekit.plugins.vad import (
    VAD,
    VADPlugin,
    VADStream,
    VADPluginOptions,
    VADEventType,
    VADSpeechData,
)
from livekit.agents.utils.voice_verification import (
    _profiler,
    analyze_voice_characteristics,
    log_audio_frame,
)

from . import onnx_model

SLOW_INFERENCE_THRESHOLD = 0.2  # late by 200ms


@dataclass
class _VADOptions:
    min_speech_duration: float
    min_silence_duration: float
    prefix_padding_duration: float
    max_buffered_speech: float
    activation_threshold: float
    sample_rate: int


class VAD(agents.vad.VAD):
    """
    Silero Voice Activity Detection (VAD) class.

    This class provides functionality to detect speech segments within audio data using the Silero VAD model.
    """  # noqa: E501

    @classmethod
    def load(
        cls,
        *,
        min_speech_duration: float = 0.05,
        min_silence_duration: float = 0.55,
        prefix_padding_duration: float = 0.5,
        max_buffered_speech: float = 60.0,
        activation_threshold: float = 0.5,
        sample_rate: Literal[8000, 16000] = 16000,
        force_cpu: bool = True,
        # deprecated
        padding_duration: NotGivenOr[float] = NOT_GIVEN,
    ) -> VAD:
        """
        Load and initialize the Silero VAD model.

        This method loads the ONNX model and prepares it for inference. When options are not provided,
        sane defaults are used.

        **Note:**
            This method is blocking and may take time to load the model into memory.
            It is recommended to call this method inside your prewarm mechanism.

        **Example:**

            ```python
            def prewarm(proc: JobProcess):
                proc.userdata["vad"] = silero.VAD.load()


            async def entrypoint(ctx: JobContext):
                vad = (ctx.proc.userdata["vad"],)
                # your agent logic...


            if __name__ == "__main__":
                cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
            ```

        Args:
            min_speech_duration (float): Minimum duration of speech to start a new speech chunk.
            min_silence_duration (float): At the end of each speech, wait this duration before ending the speech.
            prefix_padding_duration (float): Duration of padding to add to the beginning of each speech chunk.
            max_buffered_speech (float): Maximum duration of speech to keep in the buffer (in seconds).
            activation_threshold (float): Threshold to consider a frame as speech.
            sample_rate (Literal[8000, 16000]): Sample rate for the inference (only 8KHz and 16KHz are supported).
            force_cpu (bool): Force the use of CPU for inference.
            padding_duration (float | None): **Deprecated**. Use `prefix_padding_duration` instead.

        Returns:
            VAD: An instance of the VAD class ready for streaming.

        Raises:
            ValueError: If an unsupported sample rate is provided.
        """  # noqa: E501
        if sample_rate not in onnx_model.SUPPORTED_SAMPLE_RATES:
            raise ValueError("Silero VAD only supports 8KHz and 16KHz sample rates")

        if is_given(padding_duration):
            logger.warning(
                "padding_duration is deprecated and will be removed in 1.5.0, use prefix_padding_duration instead",  # noqa: E501
            )
            prefix_padding_duration = padding_duration

        session = onnx_model.new_inference_session(force_cpu)
        opts = _VADOptions(
            min_speech_duration=min_speech_duration,
            min_silence_duration=min_silence_duration,
            prefix_padding_duration=prefix_padding_duration,
            max_buffered_speech=max_buffered_speech,
            activation_threshold=activation_threshold,
            sample_rate=sample_rate,
        )
        return cls(session=session, opts=opts)

    def __init__(
        self,
        *,
        session: onnxruntime.InferenceSession,
        opts: _VADOptions,
    ) -> None:
        super().__init__(capabilities=agents.vad.VADCapabilities(update_interval=0.032))
        self._onnx_session = session
        self._opts = opts
        self._streams = weakref.WeakSet[VADStream]()

    def stream(self) -> VADStream:
        """
        Create a new VADStream for processing audio data.

        Returns:
            VADStream: A stream object for processing audio input and detecting speech.
        """
        stream = VADStream(
            self,
            self._opts,
            onnx_model.OnnxModel(
                onnx_session=self._onnx_session, sample_rate=self._opts.sample_rate
            ),
        )
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        min_speech_duration: NotGivenOr[float] = NOT_GIVEN,
        min_silence_duration: NotGivenOr[float] = NOT_GIVEN,
        prefix_padding_duration: NotGivenOr[float] = NOT_GIVEN,
        max_buffered_speech: NotGivenOr[float] = NOT_GIVEN,
        activation_threshold: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        """
        Update the VAD options.

        This method allows you to update the VAD options after the VAD object has been created.

        Args:
            min_speech_duration (float): Minimum duration of speech to start a new speech chunk.
            min_silence_duration (float): At the end of each speech, wait this duration before ending the speech.
            prefix_padding_duration (float): Duration of padding to add to the beginning of each speech chunk.
            max_buffered_speech (float): Maximum duration of speech to keep in the buffer (in seconds).
            activation_threshold (float): Threshold to consider a frame as speech.
        """  # noqa: E501
        if is_given(min_speech_duration):
            self._opts.min_speech_duration = min_speech_duration
        if is_given(min_silence_duration):
            self._opts.min_silence_duration = min_silence_duration
        if is_given(prefix_padding_duration):
            self._opts.prefix_padding_duration = prefix_padding_duration
        if is_given(max_buffered_speech):
            self._opts.max_buffered_speech = max_buffered_speech
        if is_given(activation_threshold):
            self._opts.activation_threshold = activation_threshold

        for stream in self._streams:
            stream.update_options(
                min_speech_duration=min_speech_duration,
                min_silence_duration=min_silence_duration,
                prefix_padding_duration=prefix_padding_duration,
                max_buffered_speech=max_buffered_speech,
                activation_threshold=activation_threshold,
            )


class VADStream(agents.vad.VADStream):
    def __init__(
        self, vad: VAD, opts: _VADOptions, model: onnx_model.OnnxModel
    ) -> None:
        super().__init__(vad)
        self._opts, self._model = opts, model
        self._loop = asyncio.get_event_loop()

        self._executor = ThreadPoolExecutor(max_workers=1)
        self._task.add_done_callback(lambda _: self._executor.shutdown(wait=False))
        self._exp_filter = utils.ExpFilter(alpha=0.35)

        self._input_sample_rate = 0
        self._speech_buffer: np.ndarray | None = None
        self._speech_buffer_max_reached = False
        self._prefix_padding_samples = 0  # (input_sample_rate)

    def update_options(
        self,
        *,
        min_speech_duration: NotGivenOr[float] = NOT_GIVEN,
        min_silence_duration: NotGivenOr[float] = NOT_GIVEN,
        prefix_padding_duration: NotGivenOr[float] = NOT_GIVEN,
        max_buffered_speech: NotGivenOr[float] = NOT_GIVEN,
        activation_threshold: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        """
        Update the VAD options.

        This method allows you to update the VAD options after the VAD object has been created.

        Args:
            min_speech_duration (float): Minimum duration of speech to start a new speech chunk.
            min_silence_duration (float): At the end of each speech, wait this duration before ending the speech.
            prefix_padding_duration (float): Duration of padding to add to the beginning of each speech chunk.
            max_buffered_speech (float): Maximum duration of speech to keep in the buffer (in seconds).
            activation_threshold (float): Threshold to consider a frame as speech.
        """  # noqa: E501
        old_max_buffered_speech = self._opts.max_buffered_speech

        if is_given(min_speech_duration):
            self._opts.min_speech_duration = min_speech_duration
        if is_given(min_silence_duration):
            self._opts.min_silence_duration = min_silence_duration
        if is_given(prefix_padding_duration):
            self._opts.prefix_padding_duration = prefix_padding_duration
        if is_given(max_buffered_speech):
            self._opts.max_buffered_speech = max_buffered_speech
        if is_given(activation_threshold):
            self._opts.activation_threshold = activation_threshold

        if self._input_sample_rate:
            assert self._speech_buffer is not None

            self._prefix_padding_samples = int(
                self._opts.prefix_padding_duration * self._input_sample_rate
            )

            self._speech_buffer.resize(
                int(self._opts.max_buffered_speech * self._input_sample_rate)
                + self._prefix_padding_samples
            )

            if self._opts.max_buffered_speech > old_max_buffered_speech:
                self._speech_buffer_max_reached = False

    @agents.utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        # Initialize detailed profiling
        inference_f32_data = np.empty(self._model.window_size_samples, dtype=np.float32)
        speech_buffer_index: int = 0

        # Track voice characteristics over time
        voice_history = []

        async for input_frame in self._input_ch:
            frame_start_ns = time.time_ns()

            if not isinstance(input_frame, rtc.AudioFrame):
                continue

            # LOG: Frame dequeue with queue analysis
            queue_depth = getattr(self._input_ch, "qsize", lambda: "unknown")()
            frame_id = log_audio_frame(
                input_frame,
                "vad_frame_dequeue",
                extra_data={
                    "queue_depth": queue_depth,
                    "queue_wait_estimate_ns": frame_start_ns
                    - getattr(input_frame, "_enqueue_time", frame_start_ns),
                },
            )

            # Comprehensive voice analysis
            voice_analysis = analyze_voice_characteristics(
                input_frame, voice_history[-5:] if voice_history else None
            )
            voice_history.append(input_frame)

            # LOG: Voice verification at VAD input
            logger.info(
                "VAD_VOICE_VERIFICATION",
                extra={
                    "frame_id": frame_id,
                    "location": "vad_input",
                    **voice_analysis.__dict__,
                    "voice_history_length": len(voice_history),
                    "timestamp_ns": frame_start_ns,
                },
            )

            # Memory profiling for frame processing
            memory_before = _profiler.snapshot_memory("vad_frame_start", frame_id)

            frames_to_process = [input_frame]
            frames_to_process.extend(self._input_ch.get_available_frames())

            for current_frame in frames_to_process:
                self._unprocessed_speech_frames.append(current_frame)
                speech_buffer_index += current_frame.samples_per_channel

            while True:
                inference_cycle_start = time.time_ns()  # moved from plan for clarity

                inference_frames = self._unprocessed_speech_frames
                available_inference_samples = speech_buffer_index

                if available_inference_samples < self._model.window_size_samples:
                    break

                # LOG: Window preparation
                window_prep_start = time.time_ns()
                # input_frame = utils.combine_frames(input_frames) # This was in the plan, but input_frames is not defined here
                inference_frame = utils.combine_frames(
                    inference_frames
                )  # Renamed from input_frame to inference_frame to avoid confusion
                window_prep_end = time.time_ns()

                logger.info(
                    "VAD_WINDOW_PREPARATION",
                    extra={
                        "frame_id": frame_id,  # This might need to be more specific if multiple windows are processed per input frame_id
                        "frames_combined": len(inference_frames),
                        "total_samples": available_inference_samples,  # This is total in buffer, not just current window
                        "prep_time_ns": window_prep_end - window_prep_start,
                        "timestamp_ns": window_prep_start,
                    },
                )

                # Voice verification on combined frame
                combined_voice_analysis = analyze_voice_characteristics(inference_frame)

                # Data format conversion
                conversion_start = time.time_ns()
                np.divide(
                    inference_frame.data[: self._model.window_size_samples],
                    np.iinfo(np.int16).max,
                    out=inference_f32_data,
                    dtype=np.float32,
                )
                conversion_end = time.time_ns()

                # LOG: Data conversion
                logger.info(
                    "VAD_DATA_CONVERSION",
                    extra={
                        "frame_id": frame_id,  # Same as above
                        "conversion_type": "int16_to_float32",
                        "samples_converted": self._model.window_size_samples,
                        "conversion_time_ns": conversion_end - conversion_start,
                        "timestamp_ns": conversion_start,
                    },
                )

                # Memory snapshot before inference
                memory_before_inference = _profiler.snapshot_memory(
                    "vad_before_inference", frame_id
                )  # Same as above

                # ONNX Model Inference - CRITICAL TIMING
                inference_start = time.time_ns()

                # LOG: Pre-inference state
                logger.info(
                    "VAD_INFERENCE_START",
                    extra={
                        "frame_id": frame_id,  # Same as above
                        "model_type": "silero_onnx",
                        "input_shape": inference_f32_data.shape,
                        "voice_confidence_input": combined_voice_analysis.confidence_score,
                        "memory_mb": memory_before_inference["traced_current_mb"],
                        "timestamp_ns": inference_start,
                    },
                )

                p = await self._loop.run_in_executor(
                    self._executor, self._model, inference_f32_data
                )

                inference_end = time.time_ns()
                inference_duration = (inference_end - inference_start) / 1_000_000

                # Memory snapshot after inference
                memory_after_inference = _profiler.snapshot_memory(
                    "vad_after_inference", frame_id
                )  # Same as above

                # Exponential filtering
                filter_start = time.time_ns()
                p_filtered = self._exp_filter.apply(exp=1.0, sample=p)
                filter_end = time.time_ns()

                # LOG: Complete inference results
                logger.info(
                    "VAD_INFERENCE_COMPLETE",
                    extra={
                        "frame_id": frame_id,  # Same as above
                        "location": "vad_inference_complete",
                        "raw_probability": round(p, 6),
                        "filtered_probability": round(p_filtered, 6),
                        "inference_latency_ms": round(inference_duration, 3),
                        "filter_time_ns": filter_end - filter_start,
                        "memory_delta_mb": memory_after_inference["traced_current_mb"]
                        - memory_before_inference["traced_current_mb"],
                        "voice_verification": combined_voice_analysis.__dict__,
                        "exceeds_threshold": p_filtered
                        >= self._opts.activation_threshold,
                        "activation_threshold": self._opts.activation_threshold,
                        "timestamp_ns": inference_end,
                    },
                )

                # Track inference performance
                if inference_duration > 10:  # Slow inference
                    logger.warning(
                        "SLOW_VAD_INFERENCE",
                        extra={
                            "frame_id": frame_id,  # Same as above
                            "inference_latency_ms": inference_duration,
                            "expected_max_ms": 10,
                            "performance_impact": "high_latency",
                            "timestamp_ns": inference_end,
                        },
                    )

                speech_buffer_index -= self._model.window_size_samples
                self._unprocessed_speech_frames = utils.trim_frames(
                    self._unprocessed_speech_frames, self._model.window_size_samples
                )

                if p_filtered >= self._opts.activation_threshold:
                    if not self._speaking:
                        self._start_speech(
                            input_frame
                        )  # This might need adjustment if input_frame is not the trigger

                    # ... more logging for speech buffer operations if needed ...
                    self._speech_frames.append(
                        input_frame
                    )  # This was not in plan, but seems relevant
                    self._frames_since_last_speech = 0
                else:
                    self._frames_since_last_speech += 1
                    if (
                        self._speaking
                        and self._frames_since_last_speech
                        * self._opts.min_silence_duration
                        * 1000
                        >= self._opts.min_silence_duration * 1000
                    ):
                        self._end_speech()

        # Cleanup remaining speech frames if any
        if self._speaking:
            self._end_speech()

    def _start_speech(self, frame: rtc.AudioFrame):
        logger.info(
            "VAD_SPEECH_START",
            extra={  # Added Log
                "frame_id": getattr(
                    frame, "_log_id", "unknown"
                ),  # Attempt to get frame_id if logged before
                "timestamp_ns": time.time_ns(),
            },
        )
        self._speaking = True

    def _end_speech(self):
        logger.info(
            "VAD_SPEECH_END",
            extra={  # Added Log
                "buffered_frames_count": len(self._speech_frames),
                "timestamp_ns": time.time_ns(),
            },
        )
        if not self._speaking:
            return
        self._speaking = False


class SileroVADPlugin(VADPlugin):
    def __init__(self, opts: VADPluginOptions = VADPluginOptions()):
        super().__init__(opts)
