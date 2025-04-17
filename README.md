# GPU eXecutor (gx)

A simple CLI tool that runs your commands once GPUs are available.

## Why gx?

Tired of manually checking `nvidia-smi` and waiting for GPUs to become free? `gx` automates this process by monitoring GPU availability and intelligently selecting GPUs with the least utilization for your task.

It can also notify you upon job start and completion, eliminating the need to constantly check your experiment's status.

> [!NOTE]
> While dedicated job schedulers like Slurm or Determined AI offer more comprehensive solutions, `gx` serves as a practical workaround, especially in environments like university labs where GPU resources might be managed directly on bare-metal machines without a scheduling system.

## Quick Start

```bash
# Install using uv (recommended) or pip
uv tool install gx-cli
# or: pip install gx-cli

# Example 1: Wait for one GPU with at least 40GB VRAM, then run training
gx -m 40 python train.py

# Example 2: Wait for two exclusive GPUs, each with 32GB VRAM, then run distributed training
gx -m 32 -n 2 -x python distributed_train.py
```

> [!TIP]
> If your command includes arguments that might conflict with `gx` (e.g., `-m`), use `--` to separate `gx` options from your command:
> `gx [gx-options] -- your_command [your-command-options]`
>
> Consider using scripts or command runners (like `just`) to simplify complex commands.

## Notification Setup

`gx` supports notifications via a configuration file located at `~/.config/gx/config.toml`.

Generate a default configuration file:

```bash
gx --config
```

Configure your preferred notification service:

```toml
[notification]
# Supported services: "bark", "telegram", "slack". Set to "" to disable.
service = "slack"
notify_on_gpu_found = true
notify_on_task_complete = true

# --- Service-specific settings ---
# Add the details for your chosen service below.

# Example for Slack:
[notification.slack]
webhook_url = "your-slack-webhook-url"

# Example for Telegram:
[notification.telegram]
bot_token = "your-bot-token"
chat_id = "your-chat-id"

# Example for Bark:
[notification.bark]
key = "your-bark-key"
# server = "https://api.day.app" # Optional: Specify a custom Bark server URL
```

For service setup instructions, refer to the official documentation:
- [Bark](https://bark.day.app/)
- [Telegram Bots](https://core.telegram.org/bots)
- [Slack Incoming Webhooks](https://api.slack.com/messaging/webhooks)
