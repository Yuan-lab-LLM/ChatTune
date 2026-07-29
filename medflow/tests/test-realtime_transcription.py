import asyncio
import json
import ssl
from pathlib import Path

import websockets
from loguru import logger


async def realtime_transcription(uri: str, audio_file: str = None, ssl_context=None):
    try:
        async with websockets.connect(uri, ssl=ssl_context) as websocket:
            logger.info(f"Connected to WebSocket server: {uri}")

            if audio_file and Path(audio_file).exists():
                logger.info(f"Reading the audio file: {audio_file}")

                with open(audio_file, "rb") as f:
                    audio_data = f.read()

                chunk_size = 8192

                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i : i + chunk_size]
                    await websocket.send(chunk)
                    logger.info(f"Audio block has been sent: {i // chunk_size + 1}")
            else:
                logger.info(f"No File {audio_file}.")

            try:
                while True:
                    response = await asyncio.wait_for(websocket.recv(), timeout=10)
                    response = json.loads(response)
                    if response["type"] == "TranscriptionEvent":
                        logger.info(
                            f"Received the transcription results: {response['data']['final_text']}"
                        )

            except asyncio.TimeoutError:
                logger.info("Timeout.")

    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Realtime Voice")
    parser.add_argument(
        "--uri", default="wss://localhost:9009/v1/realtime/transcriptions"
    )
    parser.add_argument("--audio", default="./data/vad_example.wav")

    parser.add_argument(
        "--cert",
        default="../web/cert.pem",
        help="Path to SSL certificate file (.pem)",
    )
    parser.add_argument(
        "--key",
        default="../web/key.pem",
        help="Path to SSL private key file (.pem)",
    )

    args = parser.parse_args()

    ssl_context = None
    if args.cert and args.key:
        try:
            ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
            ssl_context.load_cert_chain(certfile=args.cert, keyfile=args.key)
            logger.info("SSL context initialized with certificate and key")
        except FileNotFoundError as e:
            logger.error(f"SSL file not found: {e}")
            exit(1)
        except ssl.SSLError as e:
            logger.error(f"SSL configuration error: {e}")
            exit(1)
    elif args.cert or args.key:
        logger.error("Both --cert and --key must be provided for SSL")
        exit(1)

    asyncio.run(realtime_transcription(args.uri, args.audio, ssl_context))
