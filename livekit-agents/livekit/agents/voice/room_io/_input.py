from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable
from typing import Generic, TypeVar, Union, cast

from typing_extensions import override

import livekit.rtc as rtc
from livekit.rtc._proto.track_pb2 import AudioTrackFeature

from ...log import logger
from ...utils import aio, log_exceptions
from ..io import AudioInput, VideoInput
from ._pre_connect_audio import PreConnectAudioHandler
from livekit.agents.utils.audio_logging import (
    log_audio_frame,
    log_channel_operation,
    log_memory_operation,
    log_processing_step,
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

        self._forward_atask: asyncio.Task | None = None
        self._tasks: set[asyncio.Task] = set()

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

    def set_participant(self, participant: rtc.Participant | str | None) -> None:
        # set_participant can be called before the participant is connected
        participant_identity = (
            participant.identity
            if isinstance(participant, rtc.Participant)
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
            if isinstance(participant, rtc.Participant)
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
        old_task: asyncio.Task | None,
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
                continue

            # LOG: Raw WebRTC frame reception
            reception_start_ns = time.time_ns()
            frame_id = log_audio_frame(
                cast(T, event.frame),
                location="webrtc_input",
                extra_data={
                    "participant": participant.identity,
                    "source": rtc.TrackSource.Name(publication.source),
                    "stream_attached": self._attached,
                },
            )

            if frame_id:
                # LOG: Frame copy operation
                copy_start_ns = time.time_ns()
                frame_copy = cast(T, event.frame)
                copy_end_ns = time.time_ns()

                log_memory_operation(
                    frame_id=frame_id,
                    location="webrtc_frame_copy",
                    operation="frame_copy",
                    size_bytes=len(event.frame.data),
                    duration_ns=copy_end_ns - copy_start_ns,
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
                    queue_size=getattr(self._data_ch, "qsize", lambda: None)(),
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
                        "send_time_ns": send_end_ns - send_start_ns,
                    },
                )

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
        old_task: asyncio.Task | None,
        stream: rtc.AudioStream,
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
                duration = 0
                frames = await self._pre_connect_audio_handler.wait_for_data(
                    publication.track.sid
                )
                for frame in self._resample_frames(frames):
                    if self._attached:
                        # LOG: Pre-connect buffer received
                        logger.info(
                            "PRECONNECT_BUFFER_RECEIVED",
                            extra={
                                "track_id": publication.track.sid,
                                "frames_count": len(frames),
                                "wait_time_ms": round(
                                    (preconnect_receive_ns - preconnect_start_ns)
                                    / 1_000_000,
                                    3,
                                ),
                                **logging_extra,
                            },
                        )

                        # LOG: Pre-connect frame processing
                        frame_id = log_audio_frame(
                            frame,
                            location="preconnect_audio",
                            extra_data={
                                "track_id": publication.track.sid,
                                "buffer_duration_ms": round(duration * 1000, 2),
                            },
                        )

                        if frame_id:
                            await self._data_ch.send(frame)
                        else:
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
            resample_start_ns = time.time_ns()

            # LOG: Input frame to resampling
            frame_id = log_audio_frame(
                frame,
                location="resampling_input",
                extra_data={
                    "input_rate": frame.sample_rate,
                    "target_rate": self._sample_rate,
                    "resampling_needed": frame.sample_rate != self._sample_rate,
                },
            )

            if frame_id:
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

                    log_processing_step(
                        frame_id=frame_id,
                        location="resampler_creation",
                        operation="create_resampler",
                        start_time_ns=resampler_create_start,
                        end_time_ns=resampler_create_end,
                        extra_data={
                            "input_rate": frame.sample_rate,
                            "output_rate": self._sample_rate,
                        },
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
                            "total_output_samples": sum(
                                f.samples_per_channel for f in resampled_frames_list
                            ),
                        },
                    )

                    # LOG: Each resampled output frame
                    for i, resampled_frame in enumerate(resampled_frames_list):
                        log_audio_frame(
                            resampled_frame,
                            location="resampling_output",
                            frame_id=f"{frame_id}_out_{i}",
                            extra_data={
                                "original_frame_id": frame_id,
                                "resampled_index": i,
                            },
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
                        extra_data={"reason": "matching_sample_rates"},
                    )
                    yield frame
            else:
                # If the initial frame wasn't logged, we still need to pass it through the resampler or yield it
                if (
                    not resampler
                    and self._sample_rate is not None
                    and frame.sample_rate != self._sample_rate
                ):
                    resampler = rtc.AudioResampler(
                        input_rate=frame.sample_rate, output_rate=self._sample_rate
                    )

                if resampler:
                    yield from resampler.push(frame)
                else:
                    yield frame

        if resampler:
            yield from resampler.flush()


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
