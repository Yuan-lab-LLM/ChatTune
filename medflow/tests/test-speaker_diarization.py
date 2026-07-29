import argparse
import asyncio
import datetime
import ssl
from pathlib import Path

import websockets
from loguru import logger


async def spk_transcriptions(uri: str, audio_file: str = None, ssl_context=None):
    if not audio_file or not Path(audio_file).exists():
        logger.error(f"Audio file not found or invalid: {audio_file}")
        return

    try:
        async with websockets.connect(
            uri,
            ssl=ssl_context,
        ) as websocket:
            logger.info(f"Successfully connected to WebSocket server: {uri}")

            chunk_size = 4 * 1024 * 1024
            with open(audio_file, "rb") as f:
                while True:
                    chunk = f.read(chunk_size)
                    logger.info(f"Audio data sent: {len(chunk)} bytes")
                    if not chunk:
                        break
                    await websocket.send(chunk)

            await websocket.send("EOF")
            logger.info("EOF marker sent, waiting for responses")

            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            output_file = Path(f"./log/output-{timestamp}.txt").resolve()

            output_file.parent.mkdir(parents=True, exist_ok=True)

            while True:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=120)
                    logger.info(f"Received response: {response}")

                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(response + "\n")
                except asyncio.TimeoutError:
                    logger.warning(
                        "Timeout waiting for response (120s), stopping reception"
                    )
                    break
                except websockets.exceptions.ConnectionClosed:
                    logger.info(
                        "Connection closed by server, finished receiving responses"
                    )
                    break

            logger.info(f"Transcription results saved to: {output_file}")

    except websockets.exceptions.WebSocketException as e:
        logger.error(f"WebSocket connection error: {str(e)}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error occurred: {str(e)}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Speaker Diarization with WebSocket (SSL supported)"
    )
    parser.add_argument(
        "--uri",
        default="wss://localhost:9010/v1/audio/speaker_diarization",
        help="WebSocket server URI (use wss:// for SSL)",
    )
    parser.add_argument(
        "--audio", default="./data/speaker_diarization.wav", help="Path to audio file"
    )
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

    asyncio.run(spk_transcriptions(args.uri, args.audio, ssl_context))
