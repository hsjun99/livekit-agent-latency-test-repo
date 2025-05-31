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
import base64
import json
import os
import time
import weakref
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import aiohttp
import httpx

import openai
from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    stt,
    utils,
)
from livekit.agents.types import (
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import AudioBuffer, is_given
from openai.types.audio import TranscriptionVerbose
from openai.types.beta.realtime.transcription_session_update_param import (
    SessionTurnDetection,
)

from .log import logger
from .models import GroqAudioModels, STTModels
from .utils import AsyncAzureADTokenProvider
from livekit.agents.utils.voice_verification import (
    _profiler,
    analyze_voice_characteristics,
    log_audio_frame,
)

# OpenAI Realtime API has a timeout of 15 mins, we'll attempt to restart the session
# before that timeout is reached
_max_session_duration = 10 * 60
# emit interim transcriptions every 0.5 seconds
_delta_transcript_interval = 0.5
SAMPLE_RATE = 24000
NUM_CHANNELS = 1


@dataclass
class _STTOptions:
    model: STTModels | str
    language: str
    detect_language: bool
    turn_detection: SessionTurnDetection
    prompt: NotGivenOr[str] = NOT_GIVEN
    noise_reduction_type: NotGivenOr[str] = NOT_GIVEN


class STT(stt.STT):
    def __init__(
        self,
        *,
        language: str = "en",
        detect_language: bool = False,
        model: STTModels | str = "gpt-4o-mini-transcribe",
        prompt: NotGivenOr[str] = NOT_GIVEN,
        turn_detection: NotGivenOr[SessionTurnDetection] = NOT_GIVEN,
        noise_reduction_type: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        client: openai.AsyncClient | None = None,
        use_realtime: bool = False,
    ):
        """
        Create a new instance of OpenAI STT.

        Args:
            language: The language code to use for transcription (e.g., "en" for English).
            detect_language: Whether to automatically detect the language.
            model: The OpenAI model to use for transcription.
            prompt: Optional text prompt to guide the transcription. Only supported for whisper-1.
            turn_detection: When using realtime transcription, this controls how model detects the user is done speaking.
                Final transcripts are generated only after the turn is over. See: https://platform.openai.com/docs/guides/realtime-vad
            noise_reduction_type: Type of noise reduction to apply. "near_field" or "far_field"
                This isn't needed when using LiveKit's noise cancellation.
            base_url: Custom base URL for OpenAI API.
            api_key: Your OpenAI API key. If not provided, will use the OPENAI_API_KEY environment variable.
            client: Optional pre-configured OpenAI AsyncClient instance.
            use_realtime: Whether to use the realtime transcription API. (default: False)
        """  # noqa: E501

        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=use_realtime, interim_results=use_realtime
            )
        )
        if detect_language:
            language = ""

        if not is_given(turn_detection):
            turn_detection = {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 600,
                "silence_duration_ms": 350,
            }

        self._opts = _STTOptions(
            language=language,
            detect_language=detect_language,
            model=model,
            prompt=prompt,
            turn_detection=turn_detection,
        )
        if is_given(noise_reduction_type):
            self._opts.noise_reduction_type = noise_reduction_type

        self._client = client or openai.AsyncClient(
            max_retries=0,
            api_key=api_key if is_given(api_key) else None,
            base_url=base_url if is_given(base_url) else None,
            http_client=httpx.AsyncClient(
                timeout=httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0),
                follow_redirects=True,
                limits=httpx.Limits(
                    max_connections=50,
                    max_keepalive_connections=50,
                    keepalive_expiry=120,
                ),
            ),
        )

        self._streams = weakref.WeakSet[SpeechStream]()
        self._session: aiohttp.ClientSession | None = None
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            max_session_duration=_max_session_duration,
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
        )

    @staticmethod
    def with_azure(
        *,
        language: str = "en",
        detect_language: bool = False,
        model: STTModels | str = "gpt-4o-mini-transcribe",
        prompt: NotGivenOr[str] = NOT_GIVEN,
        turn_detection: NotGivenOr[SessionTurnDetection] = NOT_GIVEN,
        noise_reduction_type: NotGivenOr[str] = NOT_GIVEN,
        azure_endpoint: str | None = None,
        azure_deployment: str | None = None,
        api_version: str | None = None,
        api_key: str | None = None,
        azure_ad_token: str | None = None,
        azure_ad_token_provider: AsyncAzureADTokenProvider | None = None,
        organization: str | None = None,
        project: str | None = None,
        base_url: str | None = None,
        use_realtime: bool = False,
        timeout: httpx.Timeout | None = None,
    ) -> STT:
        """
        Create a new instance of Azure OpenAI STT.

        This automatically infers the following arguments from their corresponding environment variables if they are not provided:
        - `api_key` from `AZURE_OPENAI_API_KEY`
        - `organization` from `OPENAI_ORG_ID`
        - `project` from `OPENAI_PROJECT_ID`
        - `azure_ad_token` from `AZURE_OPENAI_AD_TOKEN`
        - `api_version` from `OPENAI_API_VERSION`
        - `azure_endpoint` from `AZURE_OPENAI_ENDPOINT`
        """  # noqa: E501

        azure_client = openai.AsyncAzureOpenAI(
            max_retries=0,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=api_version,
            api_key=api_key,
            azure_ad_token=azure_ad_token,
            azure_ad_token_provider=azure_ad_token_provider,
            organization=organization,
            project=project,
            base_url=base_url,
            timeout=(
                timeout
                if timeout
                else httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0)
            ),
        )  # type: ignore

        return STT(
            language=language,
            detect_language=detect_language,
            model=model,
            prompt=prompt,
            turn_detection=turn_detection,
            noise_reduction_type=noise_reduction_type,
            client=azure_client,
            use_realtime=use_realtime,
        )

    @staticmethod
    def with_groq(
        *,
        model: GroqAudioModels | str = "whisper-large-v3-turbo",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        client: openai.AsyncClient | None = None,
        language: str = "en",
        detect_language: bool = False,
        prompt: NotGivenOr[str] = NOT_GIVEN,
    ) -> STT:
        """
        Create a new instance of Groq STT.

        ``api_key`` must be set to your Groq API key, either using the argument or by setting
        the ``GROQ_API_KEY`` environmental variable.
        """
        groq_api_key = api_key if is_given(api_key) else os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("Groq API key is required")

        if not is_given(base_url):
            base_url = "https://api.groq.com/openai/v1"

        return STT(
            model=model,
            api_key=groq_api_key,
            base_url=base_url,
            client=client,
            language=language,
            detect_language=detect_language,
            prompt=prompt,
            use_realtime=False,
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        if is_given(language):
            self._opts.language = language
        stream = SpeechStream(
            stt=self,
            pool=self._pool,
            conn_options=conn_options,
        )
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        model: NotGivenOr[STTModels | GroqAudioModels | str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        detect_language: NotGivenOr[bool] = NOT_GIVEN,
        prompt: NotGivenOr[str] = NOT_GIVEN,
        turn_detection: NotGivenOr[SessionTurnDetection] = NOT_GIVEN,
        noise_reduction_type: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """
        Update the options for the speech stream. Most options are updated at the
        connection level. SpeechStreams will be recreated when options are updated.

        Args:
            language: The language to transcribe in.
            detect_language: Whether to automatically detect the language.
            model: The model to use for transcription.
            prompt: Optional text prompt to guide the transcription. Only supported for whisper-1.
            turn_detection: When using realtime, this controls how model detects the user is done speaking.
            noise_reduction_type: Type of noise reduction to apply. "near_field" or "far_field"
        """  # noqa: E501
        if is_given(model):
            self._opts.model = model
        if is_given(language):
            self._opts.language = language
        if is_given(detect_language):
            self._opts.detect_language = detect_language
            self._opts.language = ""
        if is_given(prompt):
            self._opts.prompt = prompt
        if is_given(turn_detection):
            self._opts.turn_detection = turn_detection
        if is_given(noise_reduction_type):
            self._opts.noise_reduction_type = noise_reduction_type

        for stream in self._streams:
            if is_given(language):
                stream.update_options(language=language)

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        prompt = self._opts.prompt if is_given(self._opts.prompt) else ""
        realtime_config: dict[str, Any] = {
            "type": "transcription_session.update",
            "session": {
                "input_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": self._opts.model,
                    "prompt": prompt,
                },
                "turn_detection": self._opts.turn_detection,
            },
        }
        if self._opts.language:
            realtime_config["session"]["input_audio_transcription"][
                "language"
            ] = self._opts.language

        if self._opts.noise_reduction_type:
            realtime_config["session"]["input_audio_noise_reduction"] = {
                "type": self._opts.noise_reduction_type
            }

        query_params: dict[str, str] = {
            "intent": "transcription",
        }
        headers = {
            "User-Agent": "LiveKit Agents",
            "Authorization": f"Bearer {self._client.api_key}",
            "OpenAI-Beta": "realtime=v1",
        }
        url = f"{str(self._client.base_url).rstrip('/')}/realtime?{urlencode(query_params)}"
        if url.startswith("http"):
            url = url.replace("http", "ws", 1)

        session = self._ensure_session()
        ws = await asyncio.wait_for(session.ws_connect(url, headers=headers), timeout)
        await ws.send_json(realtime_config)
        return ws

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        await ws.close()

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        try:
            if is_given(language):
                self._opts.language = language
            data = rtc.combine_audio_frames(buffer).to_wav_bytes()
            prompt = (
                self._opts.prompt if is_given(self._opts.prompt) else openai.NOT_GIVEN
            )

            format = "json"
            if self._opts.model == "whisper-1":
                # verbose_json returns language and other details, only supported for whisper-1
                format = "verbose_json"

            resp = await self._client.audio.transcriptions.create(
                file=(
                    "file.wav",
                    data,
                    "audio/wav",
                ),
                model=self._opts.model,  # type: ignore
                language=self._opts.language,
                prompt=prompt,
                response_format=format,
                timeout=httpx.Timeout(30, connect=conn_options.timeout),
            )

            sd = stt.SpeechData(text=resp.text, language=self._opts.language)
            if isinstance(resp, TranscriptionVerbose) and resp.language:
                sd.language = resp.language

            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[sd],
            )

        except openai.APITimeoutError:
            raise APITimeoutError() from None
        except openai.APIStatusError as e:
            raise APIStatusError(
                e.message,
                status_code=e.status_code,
                request_id=e.request_id,
                body=e.body,
            ) from None
        except Exception as e:
            raise APIConnectionError() from e


class SpeechStream(stt.SpeechStream):
    def __init__(
        self,
        *,
        stt: STT,
        conn_options: APIConnectOptions,
        pool: utils.ConnectionPool[aiohttp.ClientWebSocketResponse],
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=SAMPLE_RATE)

        self._pool = pool
        self._language = stt._opts.language
        self._request_id = ""
        self._reconnect_event = asyncio.Event()

    def update_options(
        self,
        *,
        language: str,
    ) -> None:
        self._language = language
        self._pool.invalidate()
        self._reconnect_event.set()

    @utils.log_exceptions(logger=logger)
    async def _run(self) -> None:
        closing_ws = False

        @utils.log_exceptions(logger=logger)
        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws

            # forward audio to OAI in chunks of 50ms
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
                samples_per_channel=SAMPLE_RATE // 20,
            )

            async for data in self._input_ch:
                frames: list[rtc.AudioFrame] = []
                if isinstance(data, rtc.AudioFrame):
                    frames.extend(audio_bstream.write(data.data.tobytes()))
                elif isinstance(data, self._FlushSentinel):
                    frames.extend(audio_bstream.flush())

                for input_audio_frame_chunk in frames:
                    frame_receive_ns = time.time_ns()

                    # Voice verification at STT input (this is now per CHUNK)
                    # The original `frame` from input_ch was already analyzed before this loop.
                    # This `voice_analysis` is for the current chunk produced by AudioByteStream.
                    voice_analysis_chunk = analyze_voice_characteristics(
                        input_audio_frame_chunk
                    )

                    # The frame_id should ideally be derived from the original frame or be unique per chunk.
                    # For simplicity, let's make it unique per chunk, perhaps linking to an original frame_id if available.
                    # Assuming `input_audio_frame_chunk` doesn't have an original `frame_id` directly, we make a new one.
                    # If `log_audio_frame` was called on the original frame, its id might be `frame_id_original`.
                    # Here, we log the chunk itself.
                    chunk_log_id_base = getattr(
                        input_audio_frame_chunk,
                        "_original_frame_id",
                        f"chunk_{time.time_ns()}",
                    )
                    frame_id_chunk = log_audio_frame(
                        input_audio_frame_chunk,
                        "openai_stt_input_chunk",
                        extra_data={
                            "provider": "openai",
                            "chunk_target_size_ms": 50,  # This was the stream's target, actual chunk may vary
                        },
                    )

                    # LOG: STT input voice verification (for the CHUNK)
                    logger.info(
                        "STT_CHUNK_INPUT_VOICE_VERIFICATION",
                        extra={  # Renamed log event for clarity
                            "frame_id": frame_id_chunk,  # Log ID of the current chunk
                            "provider": "openai",
                            **voice_analysis_chunk.__dict__,
                            "should_transmit": voice_analysis_chunk.is_human_voice,
                            "timestamp_ns": frame_receive_ns,
                        },
                    )

                    # Only process if likely human voice (for the CHUNK)
                    if not voice_analysis_chunk.is_human_voice:
                        logger.info(
                            "STT_CHUNK_REJECTED",
                            extra={  # Renamed log event
                                "frame_id": frame_id_chunk,
                                "reason": "not_human_voice_chunk",
                                "confidence": voice_analysis_chunk.confidence_score,
                                "rejection_reasons": voice_analysis_chunk.rejection_reasons,
                                "timestamp_ns": frame_receive_ns,
                            },
                        )
                        continue

                    # AudioByteStream chunking with detailed logging - This section is effectively done above by audio_bstream.write/
                    # The `chunks` variable from the plan isn't directly applicable here as `audio_bstream.write` already gives chunks.
                    # The following STT_CHUNKING_OPERATION log might be redundant or need rethinking.
                    # For now, I'll remove the STT_CHUNKING_OPERATION log as the chunking happens implicitly with audio_bstream.

                    # The loop `for chunk_data_bytes in chunks:` from previous attempt is now `for input_audio_frame_chunk in frames:`
                    # chunk_start_ns = time.time_ns() # Already have frame_receive_ns for chunk processing start
                    # chunk_counter += 1 # Not strictly needed if frame_id_chunk is unique

                    # chunk_id = f"{frame_id}_chunk_{chunk_counter}" # Using frame_id_chunk now

                    # Data serialization
                    serialization_start = time.time_ns()
                    frame_data_bytes = (
                        input_audio_frame_chunk.data.tobytes()
                    )  # Correctly get bytes from the chunk AudioFrame
                    base64_data = base64.b64encode(frame_data_bytes).decode()
                    serialization_end = time.time_ns()

                    # LOG: Data serialization
                    logger.info(
                        "STT_DATA_SERIALIZATION",
                        extra={
                            "chunk_id": frame_id_chunk,  # Use the chunk's log ID
                            "raw_bytes": len(frame_data_bytes),
                            "base64_bytes": len(base64_data),
                            "compression_ratio": (
                                len(base64_data) / len(frame_data_bytes)
                                if len(frame_data_bytes) > 0
                                else 0
                            ),
                            "serialization_time_ns": serialization_end
                            - serialization_start,
                            "timestamp_ns": serialization_start,
                        },
                    )

                    # Network transmission
                    transmission_start = time.time_ns()

                    try:
                        # Memory snapshot before network send
                        memory_before_send = _profiler.snapshot_memory(
                            "stt_before_send", frame_id_chunk
                        )

                        await self._conn_options.conn.send(  # This was self.conn in plan, but context is SpeechStream, so _conn_options.conn
                            rtc.ChatMessage(
                                message_id=utils.shortuuid(),
                                message=json.dumps(
                                    {
                                        "type": "input_audio_buffer.append",
                                        "audio": base64_data,
                                    }
                                ),
                            )
                        )

                        transmission_end = time.time_ns()
                        memory_after_send = _profiler.snapshot_memory(
                            "stt_after_send", frame_id_chunk
                        )

                        # LOG: Successful network transmission
                        logger.info(
                            "STT_NETWORK_TRANSMISSION",
                            extra={
                                "chunk_id": frame_id_chunk,
                                "provider": "openai",
                                "data_size_bytes": len(base64_data),
                                "transmission_latency_ms": round(
                                    (transmission_end - transmission_start) / 1_000_000,
                                    3,
                                ),
                                "memory_delta_mb": memory_after_send[
                                    "traced_current_mb"
                                ]
                                - memory_before_send["traced_current_mb"],
                                "voice_confidence": voice_analysis_chunk.confidence_score,  # Confidence of the current CHUNK
                                "network_success": True,
                                "timestamp_ns": transmission_end,
                            },
                        )

                    except Exception as e:
                        transmission_failed_ns = time.time_ns()

                        logger.error(
                            "STT_NETWORK_FAILURE",
                            extra={
                                "chunk_id": frame_id_chunk,
                                "provider": "openai",
                                "error": str(e),
                                "error_type": type(e).__name__,
                                "attempted_size_bytes": len(base64_data),
                                "transmission_attempt_time_ns": transmission_failed_ns
                                - transmission_start,
                                "timestamp_ns": transmission_failed_ns,
                            },
                        )

        @utils.log_exceptions(logger=logger)
        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws
            current_text = ""
            last_interim_at: float = 0
            connected_at = time.time()
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing_ws:  # close is expected, see SpeechStream.aclose
                        return

                    # this will trigger a reconnection, see the _run loop
                    raise APIStatusError(
                        message="OpenAI Realtime STT connection closed unexpectedly"
                    )

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected OpenAI message type %s", msg.type)
                    continue

                try:
                    data = json.loads(msg.data)
                    msg_type = data.get("type")
                    if msg_type == "conversation.item.input_audio_transcription.delta":
                        delta = data.get("delta", "")
                        if delta:
                            current_text += delta
                            if (
                                time.time() - last_interim_at
                                > _delta_transcript_interval
                            ):
                                self._event_ch.send_nowait(
                                    stt.SpeechEvent(
                                        type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                                        alternatives=[
                                            stt.SpeechData(
                                                text=current_text,
                                                language=self._language,
                                            )
                                        ],
                                    )
                                )
                                last_interim_at = time.time()
                    elif (
                        msg_type
                        == "conversation.item.input_audio_transcription.completed"
                    ):
                        current_text = ""
                        transcript = data.get("transcript", "")
                        if transcript:
                            self._event_ch.send_nowait(
                                stt.SpeechEvent(
                                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                                    alternatives=[
                                        stt.SpeechData(
                                            text=transcript,
                                            language=self._language,
                                        )
                                    ],
                                )
                            )
                        # restart session if needed
                        if time.time() - connected_at > _max_session_duration:
                            logger.info("resetting Realtime STT session due to timeout")
                            self._pool.remove(ws)
                            self._reconnect_event.set()
                            return

                except Exception:
                    logger.exception("failed to process OpenAI message")

        while True:
            async with self._pool.connection(timeout=self._conn_options.timeout) as ws:
                tasks = [
                    asyncio.create_task(send_task(ws)),
                    asyncio.create_task(recv_task(ws)),
                ]
                tasks_group = asyncio.gather(*tasks)
                wait_reconnect_task = asyncio.create_task(self._reconnect_event.wait())
                try:
                    done, _ = await asyncio.wait(
                        (tasks_group, wait_reconnect_task),
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    # propagate exceptions from completed tasks
                    for task in done:
                        if task != wait_reconnect_task:
                            task.result()

                    if wait_reconnect_task not in done:
                        break

                    self._reconnect_event.clear()
                finally:
                    await utils.aio.gracefully_cancel(*tasks, wait_reconnect_task)
                    await tasks_group
