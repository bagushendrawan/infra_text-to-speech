from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SpeechItem(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: str = Field(min_length=1)
    language: str = Field(min_length=1)
    speaker_name: str = Field(min_length=1)
    instruct: str = ""

    @field_validator("text", "language", "speaker_name", "instruct")
    @classmethod
    def strip_fields(cls, value, info):
        value = value.strip()
        if not value and info.field_name != "instruct":
            raise ValueError("Must not be blank")
        return value


class AudioResult(BaseModel):
    index: int = Field(ge=0)
    sample_rate: int = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    media_type: Literal["audio/wav"]
    audio_base64: str


class SpeechResponse(BaseModel):
    model: str
    results: list[AudioResult]
