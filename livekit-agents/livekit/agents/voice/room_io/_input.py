from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable
from typing import Any, Generic, TypeVar, Union, cast
import time
import logging

from typing_extensions import override

import livekit.rtc as rtc
from livekit.rtc._proto.track_pb2 import AudioTrackFeature

from ...log import logger
from ...utils import aio, log_exceptions
from ..io import AudioInput, VideoInput
from ._pre_connect_audio import PreConnectAudioHandler
from livekit.agents.utils.voice_verification import (
    _profiler,
    analyze_voice_characteristics,
)

T = TypeVar("T", bound=Union[rtc.AudioFrame, rtc.VideoFrame])


class _ParticipantInputStream(Generic[T], ABC):
    """
    A stream that dynamically transitions between new audio and video feeds from a connected
    participant, seamlessly switching to a different stream when the linked participant changes.
    """

    def __init__(
        self,
        room: rtc.Room,
        *,
        track_source: rtc.TrackSource.ValueType | list[rtc.TrackSource.ValueType],
    ) -> None:
        self._room = room
        self._accepted_sources = (
            {track_source}
            if isinstance(track_source, rtc.TrackSource.ValueType)
            else set(track_source)
        )

        self._data_ch = aio.Chan[T]()
        self._publication: rtc.RemoteTrackPublication | None = None
        self._stream: rtc.VideoStream | rtc.AudioStream | None = None
        self._participant_identity: str | None = None
        self._attached = True

        self._forward_atask: asyncio.Task[None] | None = None
        self._tasks: set[asyncio.Task[Any]] = set()

        self._room.on("track_subscribed", self._on_track_available)
        self._room.on("track_unpublished", self._on_track_unavailable)

    async def __anext__(self) -> T:
        return await self._data_ch.__anext__()

    def __aiter__(self) -> AsyncIterator[T]:
        return self

    @property
    def publication_source(self) -> rtc.TrackSource.ValueType:
        if not self._publication:
            return rtc.TrackSource.SOURCE_UNKNOWN
        return self._publication.source

    def on_attached(self) -> None:
        logger.debug(
            "input stream attached",
            extra={
                "participant": self._participant_identity,
                "source": rtc.TrackSource.Name(self.publication_source),
                "accepted_sources": [
                    rtc.TrackSource.Name(source) for source in self._accepted_sources
                ],
            },
        )
        self._attached = True

    def on_detached(self) -> None:
        logger.debug(
            "input stream detached",
            extra={
                "participant": self._participant_identity,
                "source": rtc.TrackSource.Name(self.publication_source),
                "accepted_sources": [
                    rtc.TrackSource.Name(source) for source in self._accepted_sources
                ],
            },
        )
        self._attached = False

    def set_participant(self, participant: rtc.RemoteParticipant | str | None) -> None:
        # set_participant can be called before the participant is connected
        participant_identity = (
            participant.identity
            if isinstance(participant, rtc.RemoteParticipant)
            else participant
        )
        if self._participant_identity == participant_identity:
            return

        self._participant_identity = participant_identity
        self._close_stream()

        if participant_identity is None:
            return

        participant = (
            participant
            if isinstance(participant, rtc.RemoteParticipant)
            else self._room.remote_participants.get(participant_identity)
        )
        if participant:
            for publication in participant.track_publications.values():
                if not publication.track:
                    continue
                self._on_track_available(publication.track, publication, participant)

    async def aclose(self) -> None:
        if self._stream:
            await self._stream.aclose()
            self._stream = None
        self._publication = None
        if self._forward_atask:
            await aio.cancel_and_wait(self._forward_atask)

        self._room.off("track_subscribed", self._on_track_available)
        self._data_ch.close()

    @log_exceptions(logger=logger)
    async def _forward_task(
        self,
        old_task: asyncio.Task[None] | None,
        stream: rtc.VideoStream | rtc.AudioStream,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if old_task:
            await aio.cancel_and_wait(old_task)

        extra = {
            "participant": participant.identity,
            "source": rtc.TrackSource.Name(publication.source),
        }
        logger.debug("start reading stream", extra=extra)
        async for event in stream:
            if not self._attached:
                # drop frames if the stream is detached
                continue
            await self._data_ch.send(cast(T, event.frame))

        logger.debug("stream closed", extra=extra)

    @abstractmethod
    def _create_stream(
        self, track: rtc.RemoteTrack
    ) -> rtc.VideoStream | rtc.AudioStream: ...

    def _close_stream(self) -> None:
        if self._stream is not None:
            task = asyncio.create_task(self._stream.aclose())
            task.add_done_callback(self._tasks.discard)
            self._tasks.add(task)
            self._stream = None
            self._publication = None

    def _on_track_available(
        self,
        track: rtc.RemoteTrack,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> bool:
        if (
            self._participant_identity != participant.identity
            or publication.source not in self._accepted_sources
            or (self._publication and self._publication.sid == publication.sid)
        ):
            return False

        self._close_stream()
        self._stream = self._create_stream(track)
        self._publication = publication
        self._forward_atask = asyncio.create_task(
            self._forward_task(
                self._forward_atask, self._stream, publication, participant
            )
        )
        return True

    def _on_track_unavailable(
        self,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if (
            not self._publication
            or self._publication.sid != publication.sid
            or participant.identity != self._participant_identity
        ):
            return

        self._close_stream()

        # subscribe to the first available track
        for publication in participant.track_publications.values():
            if publication.track is None:
                continue
            if self._on_track_available(publication.track, publication, participant):
                return


class _ParticipantAudioInputStream(_ParticipantInputStream[rtc.AudioFrame], AudioInput):
    def __init__(
        self,
        room: rtc.Room,
        *,
        sample_rate: int,
        num_channels: int,
        noise_cancellation: rtc.NoiseCancellationOptions | None,
        pre_connect_audio_handler: PreConnectAudioHandler | None,
    ) -> None:
        _ParticipantInputStream.__init__(
            self, room=room, track_source=rtc.TrackSource.SOURCE_MICROPHONE
        )
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._noise_cancellation = noise_cancellation
        self._pre_connect_audio_handler = pre_connect_audio_handler

    @override
    def _create_stream(self, track: rtc.Track) -> rtc.AudioStream:
        return rtc.AudioStream.from_track(
            track=track,
            sample_rate=self._sample_rate,
            num_channels=self._num_channels,
            noise_cancellation=self._noise_cancellation,
        )

    @override
    async def _forward_task(
        self,
        old_task: asyncio.Task[None] | None,
        stream: rtc.AudioStream,  # type: ignore[override]
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if old_task:
            await aio.cancel_and_wait(old_task)

        if (
            self._pre_connect_audio_handler
            and publication.track
            and AudioTrackFeature.TF_PRECONNECT_BUFFER in publication.audio_features
        ):
            logging_extra = {
                "track_id": publication.track.sid,
                "participant": participant.identity,
            }
            try:
                duration: float = 0
                frames = await self._pre_connect_audio_handler.wait_for_data(
                    publication.track.sid
                )
                for frame in self._resample_frames(frames):
                    if self._attached:
                        await self._data_ch.send(frame)
                        duration += frame.duration
                if frames:
                    logger.debug(
                        "pre-connect audio buffer pushed",
                        extra={"duration": duration, **logging_extra},
                    )

            except asyncio.TimeoutError:
                logger.warning(
                    "timeout waiting for pre-connect audio buffer",
                    extra=logging_extra,
                )

            except Exception as e:
                logger.error(
                    "error reading pre-connect audio buffer",
                    extra=logging_extra,
                    exc_info=e,
                )

        await super()._forward_task(old_task, stream, publication, participant)

        # push a silent frame to flush the stt final result if any
        silent_samples = int(self._sample_rate * 0.5)
        await self._data_ch.send(
            rtc.AudioFrame(
                b"\x00\x00" * silent_samples,
                sample_rate=self._sample_rate,
                num_channels=self._num_channels,
                samples_per_channel=silent_samples,
            )
        )

    def _resample_frames(
        self, frames: Iterable[rtc.AudioFrame]
    ) -> Iterable[rtc.AudioFrame]:
        resampler: rtc.AudioResampler | None = None

        for frame in frames:
            conversion_start_ns = time.time_ns()

            # LOG: Input frame analysis
            frame_id = log_audio_frame(frame, "resampling_input")

            # Voice verification on input
            voice_analysis = analyze_voice_characteristics(frame)

            # Memory snapshot before resampling
            memory_before = _profiler.snapshot_memory("resample_start", frame_id)

            if (
                not resampler
                and self._sample_rate is not None
                and frame.sample_rate != self._sample_rate
            ):

                # LOG: Resampler creation
                resampler_create_start = time.time_ns()
                resampler = rtc.AudioResampler(
                    input_rate=frame.sample_rate, output_rate=self._sample_rate
                )
                resampler_create_end = time.time_ns()

                logger.info(
                    "RESAMPLER_CREATED",
                    extra={
                        "frame_id": frame_id,
                        "input_rate": frame.sample_rate,
                        "output_rate": self._sample_rate,
                        "creation_time_ns": resampler_create_end
                        - resampler_create_start,
                        "timestamp_ns": resampler_create_start,
                    },
                )

            if resampler:
                # LOG: Before resampling operation
                resample_start_ns = time.time_ns()

                # Memory allocation for resampling
                memory_resample_start = _profiler.snapshot_memory(
                    "resample_operation", frame_id
                )

                resampled_frames_list = list(resampler.push(frame))

                resample_end_ns = time.time_ns()
                memory_resample_end = _profiler.snapshot_memory(
                    "resample_complete", frame_id
                )

                # LOG: Resampling operation details
                logger.info(
                    "RESAMPLING_OPERATION",
                    extra={
                        "frame_id": frame_id,
                        "input_samples": frame.samples_per_channel,
                        "output_frames_count": len(resampled_frames_list),
                        "total_output_samples": sum(
                            f.samples_per_channel for f in resampled_frames_list
                        ),
                        "operation_time_ns": resample_end_ns - resample_start_ns,
                        "memory_delta_mb": memory_resample_end["traced_current_mb"]
                        - memory_resample_start["traced_current_mb"],
                        "timestamp_ns": resample_start_ns,
                    },
                )

                # LOG: Each resampled frame
                for i, resampled_frame in enumerate(resampled_frames_list):
                    output_frame_id = log_audio_frame(
                        resampled_frame,
                        "resampling_output",
                        frame_id=f"{frame_id}_resampled_{i}",
                    )

                    # Voice verification on output
                    output_voice_analysis = analyze_voice_characteristics(
                        resampled_frame
                    )

                    # LOG: Voice analysis comparison
                    logger.info(
                        "RESAMPLING_VOICE_COMPARISON",
                        extra={
                            "original_frame_id": frame_id,
                            "resampled_frame_id": output_frame_id,
                            "input_voice_confidence": voice_analysis.confidence_score,
                            "output_voice_confidence": output_voice_analysis.confidence_score,
                            "voice_quality_preserved": abs(
                                voice_analysis.confidence_score
                                - output_voice_analysis.confidence_score
                            )
                            < 0.1,
                            "input_f0": voice_analysis.fundamental_freq,
                            "output_f0": output_voice_analysis.fundamental_freq,
                            "timestamp_ns": time.time_ns(),
                        },
                    )

                    yield resampled_frame
            else:
                # No resampling needed
                passthrough_end_ns = time.time_ns()

                logger.info(
                    "RESAMPLING_PASSTHROUGH",
                    extra={
                        "frame_id": frame_id,
                        "reason": "matching_sample_rates",
                        "sample_rate": frame.sample_rate,
                        "passthrough_time_ns": passthrough_end_ns - conversion_start_ns,
                        "timestamp_ns": conversion_start_ns,
                    },
                )

                yield frame


class _ParticipantVideoInputStream(_ParticipantInputStream[rtc.VideoFrame], VideoInput):
    def __init__(self, room: rtc.Room) -> None:
        _ParticipantInputStream.__init__(
            self,
            room=room,
            track_source=[
                rtc.TrackSource.SOURCE_CAMERA,
                rtc.TrackSource.SOURCE_SCREENSHARE,
            ],
        )

    @override
    def _create_stream(self, track: rtc.Track) -> rtc.VideoStream:
        return rtc.VideoStream.from_track(track=track)


def log_audio_frame(
    frame: rtc.AudioFrame, location: str, frame_id: str = None, extra_data: dict = None
) -> str:
    """Helper function to log audio frame details and return a frame_id."""
    if frame_id is None:
        frame_id = f"frame_{time.time_ns()}"

    log_data = {
        "frame_id": frame_id,
        "location": location,
        "sample_rate": frame.sample_rate,
        "num_channels": frame.num_channels,
        "samples_per_channel": frame.samples_per_channel,
        "duration_ms": (
            frame.samples_per_channel / frame.sample_rate * 1000
            if frame.sample_rate > 0 and frame.samples_per_channel > 0
            else 0
        ),
        "timestamp_ns": time.time_ns(),
    }
    if extra_data:
        log_data.update(extra_data)

    logger.info("AUDIO_FRAME_LOG", extra=log_data)
    return frame_id


StreamT = TypeVar("StreamT", bound=AsyncIterator[rtc.AudioFrameEvent])


async def _forward_task(self, stream: StreamT) -> None:
    # Memory snapshot before processing
    memory_before = _profiler.snapshot_memory("webrtc_forward_start", "session")
    send_duration_ns = 0

    async for event in stream:
        packet_start_ns = time.time_ns()
        send_end_ns = packet_start_ns

        if not isinstance(event, rtc.AudioFrameEvent):
            continue

        if not self._attached:
            # LOG: Dropped frame due to detachment
            logger.info(
                "FRAME_DROPPED",
                extra={
                    "reason": "stream_detached",
                    "timestamp_ns": packet_start_ns,
                    "frame_size": len(event.frame.data) if event.frame else 0,
                },
            )
            continue

        # LOG: Raw packet reception
        frame_id = log_audio_frame(event.frame, "webrtc_packet_received")

        # Verify human voice
        voice_analysis = analyze_voice_characteristics(event.frame)

        # LOG: Voice verification results
        logger.info(
            "VOICE_VERIFICATION",
            extra={
                "frame_id": frame_id,
                "location": "webrtc_input",
                "is_human_voice": voice_analysis.is_human_voice,
                "confidence": voice_analysis.confidence_score,
                "fundamental_freq": voice_analysis.fundamental_freq,
                "formant_f1": voice_analysis.formant_f1,
                "formant_f2": voice_analysis.formant_f2,
                "rejection_reasons": voice_analysis.rejection_reasons,
                "timestamp_ns": packet_start_ns,
            },
        )

        # Memory allocation for frame copy
        copy_start_ns = time.time_ns()
        frame_copy = cast(T, event.frame)  # This may involve memory allocation
        copy_end_ns = time.time_ns()

        # LOG: Memory operation
        logger.info(
            "MEMORY_OPERATION",
            extra={
                "frame_id": frame_id,
                "operation": "frame_copy",
                "location": "webrtc_forward",
                "duration_ns": copy_end_ns - copy_start_ns,
                "size_bytes": len(event.frame.data),
                "timestamp_ns": copy_start_ns,
            },
        )

        # Channel send operation
        send_start_ns = time.time_ns()
        try:
            await self._data_ch.send(frame_copy)
            send_end_ns = time.time_ns()

            # LOG: Successful channel send
            logger.info(
                "CHANNEL_OPERATION",
                extra={
                    "frame_id": frame_id,
                    "operation": "async_send",
                    "location": "webrtc_to_room_io",
                    "duration_ns": send_end_ns - send_start_ns,
                    "queue_size": getattr(self._data_ch, "qsize", lambda: "unknown")(),
                    "timestamp_ns": send_start_ns,
                },
            )

        except asyncio.QueueFull:
            send_end_ns = time.time_ns()
            logger.warning(
                "CHANNEL_QUEUE_FULL",
                extra={
                    "frame_id": frame_id,
                    "location": "webrtc_forward",
                    "queue_size": getattr(self._data_ch, "qsize", lambda: "unknown")(),
                    "timestamp_ns": time.time_ns(),
                },
            )
        except Exception as e:
            send_end_ns = time.time_ns()
            logger.error(
                "CHANNEL_SEND_ERROR",
                extra={
                    "frame_id": frame_id,
                    "error": str(e),
                    "timestamp_ns": time.time_ns(),
                },
            )

        packet_end_ns = time.time_ns()
        send_duration_ns = send_end_ns - send_start_ns

        # LOG: Complete packet processing time
        logger.info(
            "PACKET_PROCESSING_COMPLETE",
            extra={
                "frame_id": frame_id,
                "location": "webrtc_forward_complete",
                "total_duration_ns": packet_end_ns - packet_start_ns,
                "copy_duration_ns": copy_end_ns - copy_start_ns,
                "send_duration_ns": send_duration_ns,
                "timestamp_ns": packet_end_ns,
            },
        )

        # Periodic memory monitoring
        if frame_id.endswith("000"):  # Every 1000th frame
            memory_current = _profiler.snapshot_memory("webrtc_periodic", frame_id)
            cpu_current = _profiler.profile_cpu_usage("webrtc_periodic", frame_id)

            logger.info(
                "PERIODIC_PROFILING",
                extra={
                    "frame_id": frame_id,
                    "memory": memory_current,
                    "cpu": cpu_current,
                },
            )
