from __future__ import annotations

import asyncio
import ctypes
from collections.abc import AsyncGenerator
from typing import Union

import aiofiles

from livekit import rtc

from ..log import logger
from .aio.utils import cancel_and_wait

# Safe audio logging imports
try:
    from .safe_audio_logging import safe_log_checkpoint, safe_timing

    SAFE_LOGGING_AVAILABLE = True
except ImportError:
    SAFE_LOGGING_AVAILABLE = False

    # Create no-op functions if logging module not available
    def safe_log_checkpoint(*args, **kwargs):
        return None

    def safe_timing(*args, **kwargs):
        from contextlib import nullcontext

        return nullcontext()


# deprecated aliases
AudioBuffer = Union[list[rtc.AudioFrame], rtc.AudioFrame]

combine_frames = rtc.combine_audio_frames
merge_frames = rtc.combine_audio_frames


def calculate_audio_duration(frames: AudioBuffer) -> float:
    """
    Calculate the total duration of audio frames.

    This function computes the total duration of audio frames in seconds.
    It accepts either a list of `rtc.AudioFrame` objects or a single `rtc.AudioFrame` object.

    Parameters:
    - frames (AudioBuffer): A list of `rtc.AudioFrame` instances or a single `rtc.AudioFrame` instance.

    Returns:
    - float: The total duration in seconds of all frames provided.
    """  # noqa: E501
    if isinstance(frames, list):
        return sum(frame.duration for frame in frames)
    else:
        return frames.duration


class AudioByteStream:
    """
    Buffer and chunk audio byte data into fixed-size frames.

    This class is designed to handle incoming audio data in bytes,
    buffering it and producing audio frames of a consistent size.
    It is mainly used to easily chunk big or too small audio frames
    into a fixed size, helping to avoid processing very small frames
    (which can be inefficient) and very large frames (which can cause
    latency or processing delays). By normalizing frame sizes, it
    facilitates consistent and efficient audio data processing.
    """

    def __init__(
        self,
        sample_rate: int,
        num_channels: int,
        samples_per_channel: int | None = None,
    ) -> None:
        """
        Initialize an AudioByteStream instance.

        Parameters:
            sample_rate (int): The audio sample rate in Hz.
            num_channels (int): The number of audio channels.
            samples_per_channel (int, optional): The number of samples per channel in each frame.
                If None, defaults to `sample_rate // 10` (i.e., 100ms of audio data).

        The constructor sets up the internal buffer and calculates the size of each frame in bytes.
        The frame size is determined by the number of channels, samples per channel, and the size
        of each sample (assumed to be 16 bits or 2 bytes).
        """
        self._sample_rate = sample_rate
        self._num_channels = num_channels

        if samples_per_channel is None:
            samples_per_channel = sample_rate // 10  # 100ms by default

        self._bytes_per_sample = num_channels * ctypes.sizeof(ctypes.c_int16)
        self._bytes_per_frame = samples_per_channel * self._bytes_per_sample
        self._buf = bytearray()

    def push(self, data: bytes) -> list[rtc.AudioFrame]:
        """
        Add audio data to the buffer and retrieve fixed-size frames.

        Parameters:
            data (bytes): The incoming audio data to buffer.

        Returns:
            list[rtc.AudioFrame]: A list of `AudioFrame` objects of fixed size.

        The method appends the incoming data to the internal buffer.
        While the buffer contains enough data to form complete frames,
        it extracts the data for each frame, creates an `AudioFrame` object,
        and appends it to the list of frames to return.

        This allows you to feed in variable-sized chunks of audio data
        (e.g., from a stream or file) and receive back a list of
        fixed-size audio frames ready for processing or transmission.
        """
        # Create temporary frame for logging input
        temp_frame = rtc.AudioFrame(
            data=data,
            sample_rate=self._sample_rate,
            num_channels=self._num_channels,
            # Calculate samples_per_channel carefully to avoid division by zero if _bytes_per_sample is 0 (though unlikely)
            samples_per_channel=(
                len(data) // self._bytes_per_sample if self._bytes_per_sample > 0 else 0
            ),
        )

        frame_id = safe_log_checkpoint(
            "audiobytestream_input",
            temp_frame,
            extra={
                "target_chunk_size_ms": round(
                    (
                        self._bytes_per_frame / self._bytes_per_sample
                        if self._bytes_per_sample > 0
                        else 0
                    )
                    / self._sample_rate
                    * 1000,
                    2,
                ),
                "buffer_fill_bytes": len(self._buf),
                "frame_size_bytes": len(data),
            },
            filter_silent=True,
        )

        with safe_timing(
            frame_id, "audiobytestream_buffer_append", extra={"size_bytes": len(data)}
        ):
            self._buf.extend(data)

        frames = []
        chunk_count = 0
        while len(self._buf) >= self._bytes_per_frame:
            chunk_count += 1
            # If the parent frame_id is None (due to initial silence filtering),
            # then chunk_frame_id should also be None.
            chunk_frame_id = (
                f"{frame_id}_chunk_{chunk_count}" if frame_id is not None else None
            )

            with safe_timing(
                chunk_frame_id,
                "audiobytestream_chunk_creation_logic",
                extra={"chunk_size_bytes_target": self._bytes_per_frame},
            ):
                frame_data = self._buf[: self._bytes_per_frame]
                self._buf = self._buf[self._bytes_per_frame :]

                chunk_frame = rtc.AudioFrame(
                    data=frame_data,
                    sample_rate=self._sample_rate,
                    num_channels=self._num_channels,
                    samples_per_channel=(
                        len(frame_data) // self._bytes_per_sample
                        if self._bytes_per_sample > 0
                        else 0
                    ),
                )

            safe_log_checkpoint(
                "audiobytestream_chunk_output",
                chunk_frame,
                frame_id=chunk_frame_id,  # This will be None if parent was None
                extra={
                    "original_frame_id": frame_id,  # This can be None
                    "chunk_number": chunk_count,
                    "buffer_remaining_bytes": len(self._buf),
                },
                filter_silent=True,
            )

            frames.append(chunk_frame)

        return frames

    write = push  # Alias for the push method.

    def flush(self) -> list[rtc.AudioFrame]:
        """
        Flush the buffer and retrieve any remaining audio data as a frame.

        Returns:
            list[rtc.AudioFrame]: A list containing any remaining `AudioFrame` objects.

        This method processes any remaining data in the buffer that does not
        fill a complete frame. If the remaining data forms a partial frame
        (i.e., its size is not a multiple of the expected sample size), a warning is
        logged and an empty list is returned. Otherwise, it returns the final
        `AudioFrame` containing the remaining data.

        Use this method when you have no more data to push and want to ensure
        that all buffered audio data has been processed.
        """
        if len(self._buf) == 0:
            return []

        if len(self._buf) % (2 * self._num_channels) != 0:
            logger.warning("AudioByteStream: incomplete frame during flush, dropping")
            return []

        return [
            rtc.AudioFrame(
                data=self._buf,
                sample_rate=self._sample_rate,
                num_channels=self._num_channels,
                samples_per_channel=len(self._buf) // 2,
            )
        ]


async def audio_frames_from_file(
    file_path: str, sample_rate: int = 48000, num_channels: int = 1
) -> AsyncGenerator[rtc.AudioFrame, None]:
    """
    Decode the audio file into rtc.AudioFrame instances and yield them as an async iterable.
    Args:
        file_path (str): The path to the audio file.
        sample_rate (int, optional): Desired sample rate. Defaults to 48000.
        num_channels (int, optional): Number of channels (1 for mono, 2 for stereo). Defaults to 1.
    Returns:
        AsyncIterable[rtc.AudioFrame]: An async iterable that yields decoded AudioFrame
    """
    from .codecs import AudioStreamDecoder

    decoder = AudioStreamDecoder(sample_rate=sample_rate, num_channels=num_channels)

    async def file_reader():
        async with aiofiles.open(file_path, mode="rb") as f:
            while True:
                chunk = await f.read(4096)
                if not chunk:
                    break

                decoder.push(chunk)

        decoder.end_input()

    reader_task = asyncio.create_task(file_reader())

    try:
        async for frame in decoder:
            yield frame

    finally:
        await cancel_and_wait(reader_task)
