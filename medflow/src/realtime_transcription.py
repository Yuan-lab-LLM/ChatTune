# Copyright (c) 2025,  IEIT SYSTEMS Co.,Ltd.  All rights reserved

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import uuid

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel
from pysilero import VADIterator

from voice.streaming_sensevoice import StreamingSenseVoice

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def args_parser():
    parser = argparse.ArgumentParser(description="ASR STREAM")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=9009)
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/workspace/models/modelscope/hub/iic/SenseVoiceSmall/",
    )
    parser.add_argument("--ssl-cert", type=str, default="../web/cert.pem")
    parser.add_argument("--ssl-key", type=str, default="../web/key.pem")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--silerovad-version", type=str, default="v5")
    parser.add_argument("--samplerate", type=int, default=16000)
    parser.add_argument("--chunk-duration", type=float, default=0.1)
    parser.add_argument("--vad-min-silence-duration-ms", type=int, default=2000)
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--language", type=str, default="zh")
    parser.add_argument("--textnorm", type=str, default=True)

    args = parser.parse_args()
    return args


args = args_parser()


app = FastAPI(
    title="REALTIME ASR API",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TranscriptionChunk(BaseModel):
    timestamps: list[int]
    raw_text: str
    final_text: str | None = None
    spk_id: int | None = None


class TranscriptionEvent(BaseModel):
    type: str = "TranscriptionEvent"
    id: int
    begin_at: float
    end_at: float | None
    data: TranscriptionChunk
    is_final: bool
    session_id: str | None = None


class VADEvent(BaseModel):
    type: str = "VADEvent"
    is_active: bool


def create_transcriber():
    try:
        sensevoice_model = StreamingSenseVoice(
            model=args.model_path,
            device=args.device,
            language=args.language,
            textnorm=True,
        )

        vad_iterator = VADIterator(
            version=args.silerovad_version,
            threshold=args.vad_threshold,
            min_silence_duration_ms=args.vad_min_silence_duration_ms,
        )

        return sensevoice_model, vad_iterator
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return None, None


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "realtime-voice-transcription-api"}


@app.websocket("/v1/realtime/transcriptions")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    session_id = str(uuid.uuid4())
    logger.info(f"Session {session_id} opened")
    sensevoice_model, vad_iterator = create_transcriber()

    audio_buffer = np.array([], dtype=np.float32)
    chunk_size = int(args.chunk_duration * args.samplerate)
    speech_count = 0
    current_audio_begin_time = 0.0
    asr_detected = False
    transcription_response: TranscriptionEvent = None
    final_text = ""
    try:
        while True:
            audio_data = await websocket.receive_bytes()

            samples = (
                np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            )

            audio_buffer = np.concatenate((audio_buffer, samples))

            while len(audio_buffer) >= chunk_size:
                chunk = audio_buffer[:chunk_size]
                audio_buffer = audio_buffer[chunk_size:]

                for speech_dict, speech_samples in vad_iterator(chunk):
                    if "start" in speech_dict:
                        sensevoice_model.reset()

                        current_audio_begin_time = float(
                            speech_dict["start"] / args.samplerate
                        )

                        if asr_detected:
                            logger.debug(
                                f"{speech_count}: VAD *NOT* end: \n{transcription_response.data.raw_text}\n{str(transcription_response.data.timestamps)}"
                            )
                            speech_count += 1

                        asr_detected = False
                        logger.debug(
                            f"{speech_count}: VAD start: {current_audio_begin_time}"
                        )

                    is_last = "end" in speech_dict

                    for res in sensevoice_model.streaming_inference(
                        speech_samples,
                        is_last,
                    ):
                        if len(res["text"]) > 0:
                            asr_detected = True

                        if asr_detected:
                            transcription_response = TranscriptionEvent (
                                id=speech_count,
                                begin_at=current_audio_begin_time,
                                end_at=None,
                                data=TranscriptionChunk(
                                    timestamps=res["timestamps"],
                                    raw_text=res["text"],
                                    final_text=final_text + res["text"],
                                ),
                                is_final=False,
                                session_id=session_id,
                            )
                            await websocket.send_json(
                                transcription_response.model_dump()
                            )

                    if is_last:
                        if asr_detected:
                            speech_count += 1
                            asr_detected = False
                            transcription_response.is_final = True
                            transcription_response.end_at = (
                                speech_dict["end"] / args.samplerate
                            )
                            final_text += transcription_response.data.raw_text
                            transcription_response.data.final_text = final_text

                            await websocket.send_json(
                                transcription_response.model_dump()
                            )
                            logger.debug(
                                f"{speech_count}: VAD end: {speech_dict['end'] / args.samplerate}\n{transcription_response.data.raw_text}\n{str(transcription_response.data.timestamps)}"
                            )
                        else:
                            logger.debug(
                                f"{speech_count}: VAD end: {speech_dict['end'] / args.samplerate}\nNo Speech"
                            )
                        await websocket.send_json(
                            VADEvent(is_active=False).model_dump()
                        )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.info(f"WebSocket error {str(e)}")
    finally:
        sensevoice_model.reset()
        del sensevoice_model
        del vad_iterator
        del audio_buffer
        logger.info(f"Session {session_id} closed")


if __name__ == "__main__":
    uvicorn.run(
        "realtime_transcription:app",
        host=args.host,
        port=args.port,
        timeout_keep_alive=30,
        workers=1,
        reload=False,
        ssl_certfile=args.ssl_cert,
        ssl_keyfile=args.ssl_key,
    )
