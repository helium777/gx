import requests
from abc import ABC, abstractmethod
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor

from .settings import (
    NotificationSettings,
    BarkSettings,
    TelegramSettings,
    SlackSettings,
)


class BaseNotifier(ABC):
    """
    Base class for all notifier implementations.
    """

    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=1)

    def _make_request(self, method: str, url: str, **kwargs) -> None:
        """Make request to a url without waiting for response."""

        def _request():
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()

        self._executor.submit(_request)

    @abstractmethod
    def send(self, title: str, message: str) -> None:
        """Send a notification with the given title and message.

        Args:
            title: Notification title
            message: Notification message content
        """
        pass


class BarkNotifier(BaseNotifier):
    def __init__(self, config: BarkSettings):
        super().__init__()
        self.base_url = config.server
        self.key = config.key

    def send(self, title: str, message: str) -> None:
        url = urljoin(self.base_url, f"{self.key}/{title}/{message}")
        self._make_request("GET", url)


class TelegramNotifier(BaseNotifier):
    def __init__(self, config: TelegramSettings):
        super().__init__()
        self.base_url = "https://api.telegram.org"
        self.bot_token = config.bot_token
        self.chat_id = config.chat_id

    def send(self, title: str, message: str) -> None:
        url = urljoin(self.base_url, f"bot{self.bot_token}/sendMessage")
        payload = {
            "chat_id": self.chat_id,
            "text": f"**{title}**\n{message}",
            "parse_mode": "MarkdownV2",
        }
        self._make_request("POST", url, json=payload)


class SlackNotifier(BaseNotifier):
    def __init__(self, config: SlackSettings):
        super().__init__()
        self.webhook_url = config.webhook_url

    def send(self, title: str, message: str) -> None:
        payload = {
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": title},
                },
                {"type": "section", "text": {"type": "mrkdwn", "text": message}},
            ]
        }
        self._make_request("POST", self.webhook_url, json=payload)


class NullNotifier(BaseNotifier):
    def send(self, title: str, message: str) -> None:
        pass


registry = {
    "bark": BarkNotifier,
    "telegram": TelegramNotifier,
    "slack": SlackNotifier,
}


def get_notifier(config: NotificationSettings) -> BaseNotifier:
    service = config.service
    if service == "":
        return NullNotifier()
    elif service not in registry:
        raise ValueError(f"Unknown notification service: {service}")

    return registry[service](getattr(config, service))
