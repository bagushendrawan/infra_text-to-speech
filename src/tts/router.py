import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, Request

from src.tts.dependencies import get_service
from src.tts.exceptions import SpeechError
from src.tts.schemas import SpeechItem, SpeechResponse

router = APIRouter()


@router.post(
    "/v1/speech",
    response_model=SpeechResponse,
    tags=["speech"],
    summary="Generate an ordered batch of base64 WAV audio",
    description="Generates one PCM16 WAV per item. The complete batch succeeds or fails together.",
    responses={
        code: {"description": description}
        for code, description in {
            413: "Output size exceeded",
            422: "Invalid input or unsupported capability",
            503: "Busy or CUDA unavailable",
            504: "Generation timeout",
            500: "Inference failure",
        }.items()
    },
)
async def speech(
    items: list[SpeechItem], request: Request, service: Annotated[object, Depends(get_service)]
):
    if not items:
        raise SpeechError(422, "Batch must not be empty")
    task = asyncio.create_task(service(items))
    try:
        while not task.done():
            await asyncio.wait({task}, timeout=0.25)
            if not task.done() and await request.is_disconnected():
                task.cancel()
                break
        return await task
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
