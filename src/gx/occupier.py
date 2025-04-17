import importlib
import sys
import time
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.console import Group
from rich.padding import Padding
from rich.text import Text
from typing import List, Any, TYPE_CHECKING
import humanize

# Type hints for static type checking only
if TYPE_CHECKING:
    import numpy as np
    import numba.cuda as cuda
    import setproctitle

console = Console()


def check_and_import(module_name: str) -> Any:
    """Check if a module is available and import it."""
    try:
        return importlib.import_module(module_name)
    except ImportError:
        console.print(
            Panel.fit(
                f"[red]Error: {module_name} is required for GPU occupation[/red]\n"
                f"You may want to install the missing dependencies by installing the `occupy` extra:\n"
                f"uv tool install 'gx\[occupy\]'",
                title="Missing Dependencies",
            )
        )
        sys.exit(1)


# Runtime imports
np = check_and_import("numpy")
cuda = check_and_import("numba.cuda")
setproctitle = check_and_import("setproctitle")


def occupy_gpu(
    memory: int,
    gpu_indices: List[int],
    process_name: str = "python train.py",
):
    """
    Occupy specified amount of GPU memory and compute resources on multiple GPUs.

    Args:
        memory: Amount of GPU memory to occupy in bytes
        gpu_indices: List of GPU indices to occupy
        process_name: Name to display in process list
    """
    # Get available GPU count
    device_count = len(cuda.gpus)

    # Validate GPU indices
    invalid_gpus = [idx for idx in gpu_indices if idx >= device_count]
    if invalid_gpus:
        console.print(f"[red]Error: Invalid GPU indices: {invalid_gpus}[/red]")
        sys.exit(1)

    # Set process name
    setproctitle.setproctitle(process_name)

    # Allocate memory for each GPU
    n_floats = memory // 4  # float32 uses 4 bytes

    memory_occupiers = {}
    for gpu_idx in gpu_indices:
        with cuda.gpus[gpu_idx]:
            memory_occupiers[gpu_idx] = cuda.device_array(n_floats, dtype=np.float32)

    # Define compute kernel
    @cuda.jit
    def busy_kernel(data):
        idx = cuda.grid(1)
        if idx < data.size:
            x = float(idx)
            for _ in range(1000):
                x = (x * x + 1.0) % 1000.0
            data[idx] = x

    # Show occupation status
    console.print()
    gpu_table = Table.grid(padding=(0, 3))
    gpu_table.add_column("GPU")
    gpu_table.add_column("Memory")

    # Add memory bar for each GPU
    for gpu_idx in sorted(gpu_indices):
        memory_str = f"{humanize.naturalsize(memory, binary=True)}"

        gpu_table.add_row(
            Text(f"GPU {gpu_idx} →", style="bold cyan"),
            Text(memory_str, style="yellow"),
        )

    instructions = "[bold yellow]Press Ctrl+C to stop the GPU occupier.[/bold yellow]"

    panel_content = Group(Padding(gpu_table, (0, 0, 1, 0)), instructions)

    console.print(
        Panel(
            panel_content,
            title="GPU Occupier Status",
            title_align="left",
            expand=False,
        )
    )

    with console.status("[cyan]Occupying GPU compute resources...[/cyan]"):
        while True:
            for gpu_idx in gpu_indices:
                with cuda.gpus[gpu_idx]:
                    # Get optimal configuration for this device
                    device = cuda.gpus[gpu_idx]

                    threads_per_block = 256

                    sm_count = device.MULTIPROCESSOR_COUNT
                    desired_blocks = sm_count * 1000
                    max_blocks_per_grid = device.MAX_GRID_DIM_X
                    blocks_per_grid = min(desired_blocks, max_blocks_per_grid)

                    busy_kernel[blocks_per_grid, threads_per_block](
                        memory_occupiers[gpu_idx]
                    )

            # Simulate I/O operations
            time.sleep(0.5)
