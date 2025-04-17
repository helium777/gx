import subprocess
import sys
import os
import time
import traceback
from typing import List
import click
from rich.console import Console
from rich.panel import Panel
import tomli_w
import humanize
from rich.table import Table
from rich.progress_bar import ProgressBar
from rich.text import Text
from typing import Dict, Any

from .notify import get_notifier
from .gpu import GpuMonitor
from . import __version__
from .settings import Settings
from .globals import CONFIG_PATH

console = Console()
settings = Settings()


@click.command(context_settings={"ignore_unknown_options": True})
@click.version_option(version=__version__, prog_name="gx")
@click.option(
    "-m", "--memory", type=float, default=None, help="Required GPU memory in GiB."
)
@click.option(
    "-x",
    "--exclusive",
    is_flag=True,
    default=settings.exclusive,
    help="Only use GPUs with no other processes.",
)
@click.option("-n", "--num-gpus", type=int, default=1, help="Number of GPUs needed.")
@click.option(
    "-i",
    "--interval",
    type=float,
    default=settings.check_interval,
    help="Check interval in seconds.",
)
@click.option(
    "--occupy",
    is_flag=True,
    default=False,
    help="Only occupy GPU resources without running a command.",
)
@click.option(
    "--config",
    is_flag=True,
    default=False,
    help="Generate a default config file.",
)
@click.argument("command", nargs=-1)
def cli(
    memory: float,
    exclusive: bool,
    num_gpus: int,
    interval: float,
    occupy: bool,
    command: List[str],
    config: bool,
):
    if config:
        _generate_config()
        return

    if not memory:
        console.print(
            "[red]Error: Required GPU memory is not specified. Use --help for more information.[/red]"
        )
        sys.exit(1)

    if occupy and command:
        console.print(
            "[red]Error: Cannot specify command when using --occupy option.[/red]"
        )
        sys.exit(1)
    elif not occupy and not command:
        console.print(
            "[red]Error: Must specify a command to run or use --occupy.[/red]"
        )
        sys.exit(1)

    gpu_monitor = GpuMonitor()
    notifier = get_notifier(settings.notification)

    with console.status(
        f"Waiting for [bold green]{num_gpus} GPU(s)[/bold green] with at least [bold green]{memory}GiB[/bold green] of memory..."
    ):
        required_memory_bytes = int(memory * 1024**3)  # Convert GB to Bytes
        gpus, available_gpu_indices = gpu_monitor.wait_for_gpu(
            required_memory=required_memory_bytes,
            num_gpus=num_gpus,
            check_interval=interval,
            exclusive=exclusive,
        )
    console.print(
        f"[green]✔︎[/green] Found [bold green]{len(available_gpu_indices)} GPU(s)[/bold green] with at least [bold green]{memory}GiB[/bold green] of memory."
    )

    # Display GPU selection
    console.print()
    _display_gpu_selection(gpus, available_gpu_indices)

    if occupy:
        if settings.notification.notify_on_gpu_found:
            notifier.send(
                title="Occupier Started",
                message=f"GPU occupier started on GPUs: {available_gpu_indices}",
            )

        from .occupier import occupy_gpu

        occupy_gpu(
            memory=required_memory_bytes,
            gpu_indices=available_gpu_indices,
        )
    else:
        command_str = " ".join(command)
        if settings.notification.notify_on_gpu_found:
            notifier.send(
                title="Task Started",
                message=f"Task [{command_str}] started on GPUs: {available_gpu_indices}",
            )

        start_time = time.time()
        success = True
        error = None

        try:
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, available_gpu_indices))
            subprocess.run(command, env=env, check=True)
        except subprocess.CalledProcessError as e:
            success = False
            error = e
            console.print(f"[red]Command failed with error: {e}[/red]")

        end_time = time.time()
        duration = end_time - start_time
        duration_str = humanize.naturaldelta(duration)

        if settings.notification.notify_on_task_complete:
            if success:
                title = "Task Completed"
                message = f"Task [{command_str}] completed in {duration_str}."
            else:
                title = "Task Failed"
                message = (
                    f"Task [{command_str}] failed in {duration_str}. Error: {error}"
                )
            notifier.send(title=title, message=message)

        if not success:
            sys.exit(1)


def _generate_config():
    """Generate a default config file."""
    if CONFIG_PATH.exists():
        console.print(f"[yellow]Config file already exists at {CONFIG_PATH}[/yellow]")
    else:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        config_dict = settings.model_dump()
        with open(CONFIG_PATH, "wb") as f:
            tomli_w.dump(config_dict, f)
        console.print(f"[green]Generated default config file at {CONFIG_PATH}[/green]")


def _display_gpu_selection(gpus: Dict[int, Dict[str, Any]], selected_gpus: List[int]):
    gpus = dict(sorted(gpus.items()))

    # Create a table for GPU information
    gpu_grid = Table.grid(padding=(0, 3))
    gpu_grid.add_column("Selected")
    gpu_grid.add_column("GPU")
    gpu_grid.add_column("Memory Usage")
    gpu_grid.add_column("Utilization")
    gpu_grid.add_column("Processes")

    for gpu_idx, gpu_info in gpus.items():
        # 1. Selected
        selected_col = Text("→" if gpu_idx in selected_gpus else "", style="bold black")

        # 2. GPU ID
        gpu_col = Text(f"GPU {gpu_idx}", style="bold cyan")

        # 3. Memory usage
        total_memory = gpu_info["total_memory"]
        free_memory = gpu_info["free_memory"]
        memory_used = total_memory - free_memory
        memory_percentage = (memory_used / total_memory) * 100
        memory_color = (
            "green"
            if memory_percentage < 33
            else "yellow"
            if memory_percentage < 66
            else "red"
        )

        memory_bar = ProgressBar(
            width=20,
            total=total_memory,
            completed=memory_used,
            complete_style=memory_color,
        )

        memory_used_gb = memory_used / 1024**3
        total_memory_gb = total_memory / 1024**3
        memory_percentage_text = Text(
            f"{memory_percentage:>3.0f}%",
            style=memory_color,
        )
        memory_usage_text = Text(
            f"{memory_used_gb:>4.1f}/{total_memory_gb:>4.1f}GiB",
            style=memory_color,
        )

        memory_col = Table.grid("" * 3, padding=(0, 1))
        memory_col.add_row(memory_bar, memory_percentage_text, memory_usage_text)

        # 4. Utilization
        util_color = (
            "green"
            if gpu_info["utilization"] < 33
            else "yellow"
            if gpu_info["utilization"] < 66
            else "red"
        )

        util_bar = ProgressBar(
            width=10,
            total=100,
            completed=gpu_info["utilization"],
            complete_style=util_color,
        )

        util_col = Table.grid("" * 2, padding=(0, 1))
        util_col.add_row(
            util_bar,
            Text(f"{gpu_info['utilization']:>3.0f}%", style=util_color),
        )

        # 5. Processes
        process_color = "green" if gpu_info["process_count"] == 0 else "yellow"
        process_col = Text(
            f"Processes: {gpu_info['process_count']}", style=process_color
        )

        # Add row to table
        gpu_grid.add_row(
            selected_col,
            gpu_col,
            memory_col,
            util_col,
            process_col,
            style="dim" if gpu_idx not in selected_gpus else "",
        )

    console.print(
        Panel(
            gpu_grid,
            title="GPU Selection",
            border_style="green",
            title_align="left",
            expand=False,
        )
    )


def main():
    """Main entry point."""
    try:
        cli()
    except Exception as e:
        console.print(f"[bold red]Unexpected error: {e}[/bold red]")
        console.print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
