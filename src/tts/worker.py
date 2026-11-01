import base64
import io
import json
import logging
import sys
from pathlib import Path

from src.tts.config import Settings
from src.tts.exceptions import SpeechError


def validate_capabilities(items, languages, speakers, instruction_supported):
    languages = {x.lower() for x in languages}
    speakers = {x.lower() for x in speakers}
    for i, item in enumerate(items):
        for field, supported in [("language", languages), ("speaker_name", speakers)]:
            value = item[field].strip().lower()
            if value not in supported:
                raise SpeechError(422, f"Item {i}: unsupported {field}: {value}")
            item[field] = value
        if item["instruct"] and not instruction_supported:
            raise SpeechError(422, f"Item {i}: configured model does not support instruct")


def encode_audio(waveforms, sample_rate):
    import soundfile as sf

    results = []
    for i, wav in enumerate(waveforms):
        if len(wav) == 0:
            raise RuntimeError("Model returned empty audio")
        buffer = io.BytesIO()
        sf.write(buffer, wav, sample_rate, format="WAV", subtype="PCM_16")
        results.append(
            dict(
                index=i,
                sample_rate=sample_rate,
                duration_seconds=len(wav) / sample_rate,
                media_type="audio/wav",
                audio_base64=base64.b64encode(buffer.getvalue()).decode("ascii"),
            )
        )
    return results


def infer(items, settings):
    logging.info("Loading inference libraries")
    import torch
    from qwen_tts import Qwen3TTSModel

    if not torch.cuda.is_available():
        raise SpeechError(503, "CUDA unavailable")
    try:
        logging.info(
            "Loading model=%s device=%s (may download weights)", settings.model, settings.device
        )
        model = Qwen3TTSModel.from_pretrained(
            settings.model,
            device_map=settings.device,
            dtype=torch.float16,
            attn_implementation="eager",
        )
        logging.info("Model loaded; validating batch")
        if model.model.tts_model_type != "custom_voice":
            raise SpeechError(422, "Configured model must be CustomVoice")
        validate_capabilities(
            items,
            model.get_supported_languages() or [],
            model.get_supported_speakers() or [],
            model.model.tts_model_size != "0b6",
        )
        logging.info("Generating audio items=%s", len(items))
        wavs, sr = model.generate_custom_voice(
            text=[x["text"] for x in items],
            language=[x["language"] for x in items],
            speaker=[x["speaker_name"] for x in items],
            instruct=[x["instruct"] for x in items],
            non_streaming_mode=True,
            max_new_tokens=settings.max_new_tokens,
        )
        logging.info("Generation complete; encoding WAV results")
        if len(wavs) != len(items):
            raise RuntimeError("Model output count mismatch")
        return dict(model=settings.model, results=encode_audio(wavs, sr))
    except torch.cuda.OutOfMemoryError as exc:
        raise SpeechError(503, "GPU out of memory") from exc


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s TTS worker pid=%(process)d %(levelname)s %(message)s",
    )
    logging.info("Worker started")
    source, target = map(Path, sys.argv[1:])
    payload = json.loads(source.read_text())
    settings = Settings(**payload["settings"])
    try:
        result = infer(payload["items"], settings)
        encoded = json.dumps(result)
        if len(encoded.encode()) > settings.max_result_bytes:
            raise SpeechError(413, "Generated result exceeds size limit")
    except SpeechError as exc:
        logging.warning("Request failed status=%s detail=%s", exc.status, exc.detail)
        encoded = json.dumps(dict(status=exc.status, error=exc.detail))
    except Exception:
        logging.exception("Speech inference failed")
        encoded = json.dumps(dict(status=500, error="Speech inference failed"))
    target.write_text(encoded, encoding="utf-8")
    logging.info("Worker result written; exiting")


if __name__ == "__main__":
    main()
