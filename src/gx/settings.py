from typing import Literal, Type, Tuple
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)
from pydantic import Field

from .globals import CONFIG_PATH


class BarkSettings(BaseSettings):
    """Configuration for Bark notification service."""

    key: str = ""
    server: str = "https://api.day.app"


class TelegramSettings(BaseSettings):
    """Configuration for Telegram notification service."""

    bot_token: str = ""
    chat_id: str = ""


class SlackSettings(BaseSettings):
    """Configuration for Slack notification service."""

    webhook_url: str = ""


class NotificationSettings(BaseSettings):
    """Configuration for notification system."""

    service: Literal["", "bark", "telegram", "slack"] = ""
    notify_on_gpu_found: bool = True
    notify_on_task_complete: bool = True
    bark: BarkSettings = Field(default_factory=BarkSettings)
    telegram: TelegramSettings = Field(default_factory=TelegramSettings)
    slack: SlackSettings = Field(default_factory=SlackSettings)


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(toml_file=CONFIG_PATH)

    check_interval: float = 1.0
    exclusive: bool = False
    notification: NotificationSettings = Field(default_factory=NotificationSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (TomlConfigSettingsSource(settings_cls),)
