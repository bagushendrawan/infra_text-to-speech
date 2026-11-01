# Text to speech

Start from this directory with `uv run python main.py` (localhost:9103).
See [shared setup and API smoke tests](../README.md).

Standalone Python 3.12 FastAPI service for Qwen3-TTS CustomVoice. Each request
loads the model into a fresh CUDA subprocess and releases GPU memory when it
exits. Model downloads remain in the Hugging Face cache. There is no audio
storage, queue, streaming, authentication, or CPU fallback.

## Run

Requires Linux, an NVIDIA CUDA GPU with sufficient free VRAM, a CUDA-compatible
PyTorch installation, libsndfile, and SoX. Install system dependencies with your
OS package manager (on Debian/Ubuntu: `libsndfile1 sox`).

```sh
uv sync --locked --extra inference
cp .env.example .env
uv run --no-sync uvicorn src.main:app --host 127.0.0.1 --port 9103 --workers 1
```

Use one API worker per assigned GPU. Admission control is per process; it does
not coordinate with other AI tools. Keep this service on a private network and
configure ingress body limits and client/proxy timeouts above TTS_TIMEOUT.
First requests may download weights and take longer than subsequent requests;
model loading still occurs on every request. `/health/live` checks the API only,
not CUDA availability or whether weights can load. `/docs` is available when
TTS_ENVIRONMENT is local or staging.

## API

```sh
curl http://127.0.0.1:9103/v1/speech \
  -H 'Content-Type: application/json' \
  -d '[{"text":"Hello!","language":"English","speaker_name":"Ryan","instruct":""}]'
```

The top-level body is an array. text, language and speaker_name are required,
nonblank strings; instruct defaults to an empty string. Unknown fields fail
validation. Languages and speakers are case-insensitive and validated against
the loaded model. Use language `Auto` for automatic language selection.
Supported languages and voices are described in the
[upstream README](https://github.com/QwenLM/Qwen3-TTS#custom-voice-generate).
Indonesian is not supported by the released models.

```json
{"model":"Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice","results":[{"index":0,"sample_rate":24000,"duration_seconds":1.2,"media_type":"audio/wav","audio_base64":"..."}]}
```

Results preserve input order, with one mono PCM16 WAV per item. Decode
`audio_base64` as ordinary base64 bytes (no data URL prefix), and write those
bytes to a `.wav` file. Sample rate and duration come from actual model output.
The whole batch succeeds or fails together.

The default 0.6B model does not support instructions. Non-empty instruct returns
422. Set TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice to enable instructions.
TTS_MODEL also accepts a local CustomVoice model directory. Base and VoiceDesign
models cannot satisfy this API and return 422.

Errors use `{"detail": ...}`: 422 invalid input/unsupported capability (including
an item index for per-item errors), 413 generated result exceeds the size limit,
503 busy/CUDA unavailable/GPU out of memory, 504 processing deadline, and 500
inference failure. Busy requests are rejected immediately. Client disconnect,
timeout, or graceful shutdown terminates the worker and cleans temporary files.
Uncatchable process termination may leave OS temporary files.

## Settings

All settings use TTS_ and can be supplied by environment or this directory's
`.env` (run commands from here). See `.env.example` for every setting.
Defaults: cuda:0; 8 items; 2,000 text characters per item; 8,000 text characters
per batch; 1,000 instruction characters per item; 600-second processing timeout;
2,048 generation tokens per item; 32 MiB encoded JSON result limit. Text and
instruction whitespace is stripped before length checks. Unsupported fields
are rejected, and CPU device values are rejected at configuration time.
Generation can be truncated at the token limit; split longer text into items
or raise the environment limit. Output limits are checked after generation and
encoding; they do not impose a hard peak-memory bound. Base64 adds roughly 33%
to the WAV size. WAV encoding, generation, and result construction run outside
the API process. Logs contain request IDs, status, and elapsed time.

## Development

```sh
uv sync --locked
uv run --no-sync pytest -q
uv run --no-sync ruff check src tests
uv run --no-sync ruff format --check src tests
uv lock --check --offline
```

API tests use dependency overrides. Worker lifecycle tests run real lightweight
subprocesses; WAV tests use a synthetic signal. Real inference is opt-in:

```sh
TTS_RUN_GPU=1 uv run --extra inference pytest -q tests/test_gpu.py
```

This covers 0.6B speech and 1.7B instruction-enabled generation. CUDA inference,
GPU capacity, and speech quality are UNVERIFIED until tested on the deployment
host. Automated smoke tests check valid audio, not perceptual instruction
compliance. The upstream package and downloaded models retain their licenses.
