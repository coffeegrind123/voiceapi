"""Run the gateway: ``python -m voiceapi_gateway`` or ``voiceapi-gateway``."""

from __future__ import annotations

import logging
import os

import uvicorn


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("VOICEAPI_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    from voiceapi_gateway.app import create_app

    uvicorn.run(
        create_app(),
        host=os.environ.get("VOICEAPI_HOST", "0.0.0.0"),  # noqa: S104 - container service
        port=int(os.environ.get("VOICEAPI_PORT", "8080")),
        proxy_headers=True,
        forwarded_allow_ips=os.environ.get("VOICEAPI_FORWARDED_ALLOW_IPS", "127.0.0.1"),
        log_level=os.environ.get("VOICEAPI_LOG_LEVEL", "info").lower(),
        timeout_keep_alive=30,
    )


if __name__ == "__main__":
    main()
