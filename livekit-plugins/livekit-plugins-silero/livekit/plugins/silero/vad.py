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

from . import onnx_model
from .log import logger
from livekit.agents.utils.audio_logging import (
    calculate_audio_metrics,
    log_audio_frame,
    log_processing_step,
)

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
    async def _main_task(self):
        inference_f32_data = np.empty(self._model.window_size_samples, dtype=np.float32)
        speech_buffer_index: int = 0

        # "pub_" means public, these values are exposed to the users through events
        pub_speaking = False
        pub_speech_duration = 0.0
        pub_silence_duration = 0.0
        pub_current_sample = 0
        pub_timestamp = 0.0

        speech_threshold_duration = 0.0
        silence_threshold_duration = 0.0

        input_frames_list = []
        inference_frames = []
        resampler: rtc.AudioResampler | None = None

        # used to avoid drift when the sample_rate ratio is not an integer
        input_copy_remaining_fract = 0.0

        extra_inference_time = 0.0

        async for input_frame_from_channel in self._input_ch:
            frame_receive_ns = time.time_ns()

            if not isinstance(input_frame_from_channel, rtc.AudioFrame):
                continue

            # LOG: VAD frame dequeue (This was missing)
            # This frame_id determines if subsequent detailed logs for this input_frame_from_channel are made.
            frame_id = log_audio_frame(
                input_frame_from_channel,  # Use the frame as received from the channel
                location="vad_frame_dequeue",
                extra_data={
                    "queue_depth": getattr(self._input_ch, "qsize", lambda: None)(),
                    "vad_type": "silero",
                },
            )

            if not self._input_sample_rate:
                self._input_sample_rate = input_frame_from_channel.sample_rate
                # alloc the buffers now that we know the input sample rate
                self._prefix_padding_samples = int(
                    self._opts.prefix_padding_duration * self._input_sample_rate
                )

                self._speech_buffer = np.empty(
                    int(self._opts.max_buffered_speech * self._input_sample_rate)
                    + self._prefix_padding_samples,
                    dtype=np.int16,
                )

                if self._input_sample_rate != self._opts.sample_rate:
                    # resampling needed: the input sample rate isn't the same as the model's
                    # sample rate used for inference
                    resampler = rtc.AudioResampler(
                        input_rate=self._input_sample_rate,
                        output_rate=self._opts.sample_rate,
                        quality=rtc.AudioResamplerQuality.QUICK,  # VAD doesn't need high quality
                    )

            elif self._input_sample_rate != input_frame_from_channel.sample_rate:
                logger.error("a frame with another sample rate was already pushed")
                continue

            assert self._speech_buffer is not None

            input_frames_list.append(input_frame_from_channel)
            if resampler is not None:
                # the resampler may have a bit of latency, but it is OK to ignore since it should be
                # negligible
                inference_frames.extend(resampler.push(input_frame_from_channel))
            else:
                inference_frames.append(input_frame_from_channel)

            # The while loop processes chunks. If frame_id is None, these logs are skipped.
            original_frame_id_for_this_input_cycle = (
                frame_id  # Capture frame_id for this cycle. Now frame_id is defined.
            )

            while True:  # Keep the original loop condition for VAD logic execution
                inference_cycle_start = time.time_ns()

                available_inference_samples = sum(
                    [frame.samples_per_channel for frame in inference_frames]
                )
                if available_inference_samples < self._model.window_size_samples:
                    break

                # This combined_input_frame is based on the current input_frames_list, used for speech buffer copy
                combined_input_frame_for_buffer = utils.combine_frames(
                    input_frames_list
                )
                # This is the frame used for actual VAD model inference (potentially resampled)
                current_inference_audio_frame = utils.combine_frames(inference_frames)

                if (
                    original_frame_id_for_this_input_cycle
                ):  # Only log if the initial frame was logged
                    # LOG: Frame combination for VAD window (This is about the inference_frame)
                    combine_start = time.time_ns()
                    combine_end = time.time_ns()

                    log_processing_step(
                        frame_id=original_frame_id_for_this_input_cycle,
                        location="vad_frame_combination",
                        operation="combine_frames_for_window",
                        start_time_ns=combine_start,
                        end_time_ns=combine_end,
                        extra_data={
                            "frames_combined_for_inference": len(
                                inference_frames
                            ),  # Clarified name
                            "window_size_samples": self._model.window_size_samples,
                            "total_samples_in_inference_frame": current_inference_audio_frame.samples_per_channel,
                        },
                    )

                    # LOG: Data format conversion
                    conversion_start = time.time_ns()
                    np.divide(
                        current_inference_audio_frame.data[
                            : self._model.window_size_samples
                        ],
                        np.iinfo(np.int16).max,
                        out=inference_f32_data,
                        dtype=np.float32,
                    )
                    conversion_end = time.time_ns()

                    log_processing_step(
                        frame_id=original_frame_id_for_this_input_cycle,
                        location="vad_data_conversion",
                        operation="int16_to_float32",
                        start_time_ns=conversion_start,
                        end_time_ns=conversion_end,
                        extra_data={
                            "samples_converted": self._model.window_size_samples,
                            "input_dtype": "int16",
                            "output_dtype": "float32",
                        },
                    )

                # CRITICAL: VAD Inference
                inference_start = time.time_ns()
                # Calculate metrics on the frame fed to the model
                input_metrics = calculate_audio_metrics(current_inference_audio_frame)

                if original_frame_id_for_this_input_cycle:
                    logger.info(
                        f"VAD_INFERENCE_START: {{'frame_id': '{original_frame_id_for_this_input_cycle}', 'location': 'vad_inference_start', 'model_type': 'silero_onnx', 'window_size': {self._model.window_size_samples}, 'input_dbfs': {input_metrics.dbfs}, 'input_has_content': {input_metrics.has_audio_content}, 'timestamp_ns': {inference_start}}}"
                    )

                p = await self._loop.run_in_executor(
                    self._executor, self._model, inference_f32_data
                )

                inference_end = time.time_ns()
                inference_duration_calc = (
                    inference_end - inference_start
                ) / 1_000_000  # Renamed to avoid clash

                filter_start = time.time_ns()
                p_filtered = self._exp_filter.apply(exp=1.0, sample=p)
                filter_end = time.time_ns()

                if original_frame_id_for_this_input_cycle:
                    logger.info(
                        f"VAD_INFERENCE_COMPLETE: {{'frame_id': '{original_frame_id_for_this_input_cycle}', 'location': 'vad_inference_complete', 'raw_probability': {round(p, 6)}, 'filtered_probability': {round(p_filtered, 6)}, 'inference_latency_ms': {round(inference_duration_calc, 3)}, 'filter_time_ns': {filter_end - filter_start}, 'exceeds_threshold': {p_filtered >= self._opts.activation_threshold}, 'activation_threshold': {self._opts.activation_threshold}, 'input_audio_metrics': {input_metrics.__dict__}, 'timestamp_ns': {inference_end}}}"
                    )

                    if inference_duration_calc > 50:
                        logger.warning(
                            f"SLOW_VAD_INFERENCE: {{'frame_id': '{original_frame_id_for_this_input_cycle}', 'inference_latency_ms': {inference_duration_calc}, 'expected_max_ms': 50, 'timestamp_ns': {inference_end}}}"
                        )

                # VAD core logic based on p_filtered (original logic from the file)
                window_duration = (
                    self._model.window_size_samples / self._opts.sample_rate
                )

                pub_current_sample += self._model.window_size_samples
                pub_timestamp += window_duration

                resampling_ratio = self._input_sample_rate / self._model.sample_rate
                to_copy = (
                    self._model.window_size_samples * resampling_ratio
                    + input_copy_remaining_fract
                )
                to_copy_int = int(to_copy)
                input_copy_remaining_fract = to_copy - to_copy_int

                available_space = len(self._speech_buffer) - speech_buffer_index
                to_copy_buffer = min(to_copy_int, available_space)
                if to_copy_buffer > 0:
                    self._speech_buffer[
                        speech_buffer_index : speech_buffer_index + to_copy_buffer
                    ] = combined_input_frame_for_buffer.data[
                        :to_copy_buffer
                    ]  # Use combined_input_frame_for_buffer
                    speech_buffer_index += to_copy_buffer
                elif not self._speech_buffer_max_reached:
                    # reached self._opts.max_buffered_speech (padding is included)
                    speech_buffer_max_reached = True
                    logger.warning(
                        "max_buffered_speech reached, ignoring further data for the current speech input"  # noqa: E501
                    )

                # The original code had `inference_duration = time.perf_counter() - start_time`
                # `start_time` was not defined. I am using `inference_duration_calc` calculated above.
                # The `extra_inference_time` logic seems to rely on this.
                current_cycle_actual_processing_duration = (
                    time.time_ns() - inference_cycle_start
                ) / 1e9  # time from start of this while iter
                extra_inference_time = max(
                    0.0,
                    extra_inference_time
                    + current_cycle_actual_processing_duration
                    - window_duration,
                )
                if (
                    current_cycle_actual_processing_duration > SLOW_INFERENCE_THRESHOLD
                ):  # SLOW_INFERENCE_THRESHOLD is 0.2s
                    logger.warning(
                        "VAD inference cycle is slower than realtime window",
                        extra={
                            "delay_ms": extra_inference_time * 1000,
                            "cycle_duration_ms": current_cycle_actual_processing_duration
                            * 1000,
                            "window_duration_ms": window_duration * 1000,
                        },
                    )

                def _reset_write_cursor():
                    nonlocal speech_buffer_index  # _speech_buffer_max_reached is an instance var, accessed via self
                    assert self._speech_buffer is not None
                    if speech_buffer_index <= self._prefix_padding_samples:
                        return
                    padding_data = self._speech_buffer[
                        speech_buffer_index
                        - self._prefix_padding_samples : speech_buffer_index
                    ]
                    self._speech_buffer_max_reached = False  # Accessed via self
                    self._speech_buffer[: self._prefix_padding_samples] = padding_data
                    speech_buffer_index = self._prefix_padding_samples

                def _copy_speech_buffer() -> rtc.AudioFrame:
                    assert self._speech_buffer is not None
                    speech_data = self._speech_buffer[:speech_buffer_index].tobytes()
                    return rtc.AudioFrame(
                        sample_rate=self._input_sample_rate,
                        num_channels=1,
                        samples_per_channel=speech_buffer_index,
                        data=speech_data,
                    )

                if pub_speaking:
                    pub_speech_duration += window_duration
                else:
                    pub_silence_duration += window_duration

                self._event_ch.send_nowait(
                    agents.vad.VADEvent(
                        type=agents.vad.VADEventType.INFERENCE_DONE,
                        samples_index=pub_current_sample,
                        timestamp=pub_timestamp,
                        silence_duration=pub_silence_duration,
                        speech_duration=pub_speech_duration,
                        probability=p_filtered,  # Use p_filtered
                        inference_duration=inference_duration_calc,  # Use calculated duration
                        frames=[
                            rtc.AudioFrame(
                                # This should be the portion of the original input corresponding to the window
                                data=combined_input_frame_for_buffer.data[
                                    :to_copy_int
                                ].tobytes(),
                                sample_rate=self._input_sample_rate,
                                num_channels=1,
                                samples_per_channel=to_copy_int,
                            )
                        ],
                        speaking=pub_speaking,
                        raw_accumulated_silence=silence_threshold_duration,
                        raw_accumulated_speech=speech_threshold_duration,
                    )
                )

                if p_filtered >= self._opts.activation_threshold:
                    speech_threshold_duration += window_duration
                    silence_threshold_duration = 0.0
                    if not pub_speaking:
                        if speech_threshold_duration >= self._opts.min_speech_duration:
                            pub_speaking = True
                            pub_silence_duration = 0.0
                            pub_speech_duration = speech_threshold_duration
                            self._event_ch.send_nowait(
                                agents.vad.VADEvent(
                                    type=agents.vad.VADEventType.START_OF_SPEECH,
                                    samples_index=pub_current_sample,
                                    timestamp=pub_timestamp,
                                    silence_duration=pub_silence_duration,
                                    speech_duration=pub_speech_duration,
                                    frames=[_copy_speech_buffer()],
                                    speaking=True,
                                )
                            )
                else:
                    silence_threshold_duration += window_duration
                    speech_threshold_duration = 0.0
                    if not pub_speaking:
                        _reset_write_cursor()
                    if (
                        pub_speaking
                        and silence_threshold_duration
                        >= self._opts.min_silence_duration
                    ):
                        pub_speaking = False
                        pub_speech_duration = 0.0
                        pub_silence_duration = silence_threshold_duration
                        self._event_ch.send_nowait(
                            agents.vad.VADEvent(
                                type=agents.vad.VADEventType.END_OF_SPEECH,
                                samples_index=pub_current_sample,
                                timestamp=pub_timestamp,
                                silence_duration=pub_silence_duration,
                                speech_duration=pub_speech_duration,
                                frames=[_copy_speech_buffer()],
                                speaking=False,
                            )
                        )
                        _reset_write_cursor()

                # remove the frames that were used for inference from the input and inference frames
                input_frames_list = (
                    []
                )  # Clear the list for the next input_frame_from_channel
                inference_frames = []

                # add the remaining data from combined_input_frame_for_buffer (original rate)
                if len(combined_input_frame_for_buffer.data) - to_copy_int > 0:
                    data = combined_input_frame_for_buffer.data[to_copy_int:].tobytes()
                    input_frames_list.append(
                        rtc.AudioFrame(
                            data=data,
                            sample_rate=self._input_sample_rate,
                            num_channels=1,
                            samples_per_channel=len(data)
                            // (
                                1 * 2
                            ),  # num_channels * bytes_per_sample (assuming 16-bit)
                        )
                    )
                # add remaining from current_inference_audio_frame (model rate)
                if (
                    len(current_inference_audio_frame.data)
                    - self._model.window_size_samples
                    > 0
                ):
                    data = current_inference_audio_frame.data[
                        self._model.window_size_samples :
                    ].tobytes()
                    inference_frames.append(
                        rtc.AudioFrame(
                            data=data,
                            sample_rate=self._opts.sample_rate,
                            num_channels=1,
                            samples_per_channel=len(data)
                            // (1 * 2),  # num_channels * bytes_per_sample
                        )
                    )
