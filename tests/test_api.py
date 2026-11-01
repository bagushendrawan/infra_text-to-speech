import pytest
from httpx import ASGITransport, AsyncClient

from src.main import create_app
from src.tts.dependencies import get_service
from src.tts.exceptions import SpeechError

ITEM = dict(text="Hello", language="English", speaker_name="Ryan")


@pytest.mark.asyncio
async def test_contract():
    app = create_app()

    async def fake(items):
        return dict(
            model="test",
            results=[
                dict(
                    index=i,
                    sample_rate=24000,
                    duration_seconds=1,
                    media_type="audio/wav",
                    audio_base64="UklGRg==",
                )
                for i in range(len(items))
            ],
        )

    app.dependency_overrides[get_service] = lambda: fake
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/v1/speech", json=[ITEM, ITEM])
        assert r.status_code == 200
        assert [x["index"] for x in r.json()["results"]] == [0, 1]
        assert r.headers["X-Request-ID"]
        for payload in [[], [dict(ITEM, text=" ")], [dict(ITEM, extra=True)], [dict(text="x")]]:
            assert (await c.post("/v1/speech", json=payload)).status_code == 422
        assert (await c.get("/health/live")).status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [413, 422, 500, 503, 504])
async def test_errors(status):
    app = create_app()

    async def fake(items):
        raise SpeechError(status, "failure")

    app.dependency_overrides[get_service] = lambda: fake
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/v1/speech", json=[ITEM])
        assert r.status_code == status
        assert r.json() == {"detail": "failure"}


@pytest.mark.asyncio
async def test_documentation():
    from src.tts.config import Settings

    for environment, status in [("local", 200), ("staging", 200), ("production", 404)]:
        app = create_app(Settings(environment=environment))
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            assert (await c.get("/docs")).status_code == status
            assert (await c.get("/openapi.json")).status_code == status
