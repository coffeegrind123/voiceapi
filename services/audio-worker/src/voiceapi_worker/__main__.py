"""Run the audio worker: ``python -m voiceapi_worker``."""

from __future__ import annotations

import logging
import os

import uvicorn


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("VOICEAPI_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    from voiceapi_worker.app import create_app

    uvicorn.run(
        create_app(),
        host=os.environ.get("VOICEAPI_WORKER_HOST", "0.0.0.0"),  # noqa: S104 - compose-internal service
        port=int(os.environ.get("VOICEAPI_WORKER_PORT", "8001")),
        log_level=os.environ.get("VOICEAPI_LOG_LEVEL", "info").lower(),
    )


if __name__ == "__main__":
    main()
