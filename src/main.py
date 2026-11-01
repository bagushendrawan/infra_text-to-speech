import logging
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.tts.config import Settings
from src.tts.exceptions import SpeechError
from src.tts.router import router
from src.tts.service import SpeechService

logger = logging.getLogger("uvicorn.error")


def create_app(settings=None):
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        app.state.service = SpeechService(settings)
        try:
            yield
        finally:
            await app.state.service.close()

    app = FastAPI(
        title="Text to speech",
        lifespan=lifespan,
        openapi_url="/openapi.json" if settings.environment in {"local", "staging"} else None,
    )
    app.include_router(router)

    @app.exception_handler(SpeechError)
    async def speech_error(request, exc):
        return JSONResponse(
            {"detail": exc.detail},
            status_code=exc.status,
            headers={"Retry-After": "5"} if exc.status == 503 else None,
        )

    @app.middleware("http")
    async def timing(request: Request, call_next):
        request_id = str(uuid4())
        start = time.monotonic()
        logger.info(
            "request_id=%s started method=%s path=%s", request_id, request.method, request.url.path
        )
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request_id=%s status=%s elapsed_ms=%.2f",
            request_id,
            response.status_code,
            (time.monotonic() - start) * 1000,
        )
        return response

    @app.get("/health/live")
    async def live():
        return {"status": "ok"}

    return app


app = create_app()
