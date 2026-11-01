from typing import Annotated

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PositiveInt = Annotated[int, Field(gt=0)]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TTS_", env_file=".env", extra="ignore")
    model: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    device: str = Field(default="cuda:0", pattern=r"^cuda(:[0-9]+)?$")
    environment: str = "local"
    max_items: PositiveInt = 8
    max_text_chars: PositiveInt = 2000
    max_total_chars: PositiveInt = 8000
    max_instruction_chars: PositiveInt = 1000
    timeout: float = Field(default=600, gt=0)
    max_new_tokens: PositiveInt = 2048
    max_result_bytes: PositiveInt = 32 * 1024 * 1024
