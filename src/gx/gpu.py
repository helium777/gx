import time
from typing import List, Optional, Dict, Any, Tuple
import pynvml


class GpuMonitor:
    def __init__(self):
        pynvml.nvmlInit()

    def __del__(self):
        pynvml.nvmlShutdown()

    def _get_gpu_info(self) -> Dict[int, Dict[str, Any]]:
        """Get comprehensive GPU information using NVML."""
        device_count = pynvml.nvmlDeviceGetCount()
        gpus = {}
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            process_count = len(processes)

            gpus[i] = {
                "free_memory": info.free,
                "total_memory": info.total,
                "utilization": utilization,
                "process_count": process_count,
            }
        return gpus

    def _find_available_gpu(
        self,
        gpu_info: Dict[int, Dict[str, Any]],
        required_memory: int,
        num_gpus: int,
        sort: bool = True,
        exclusive: bool = False,
    ) -> Optional[List[int]]:
        """Find GPUs with sufficient free memory.

        Args:
            gpu_info: Dictionary containing information about each GPU.
            required_memory: Required memory in bytes for each GPU
            num_gpus: Number of GPUs needed
            sort: Whether to sort the GPUs by utilization and free memory
            exclusive: Whether to only return GPUs with no running processes

        Returns:
            List of GPU indices if found, None otherwise
        """
        gpus = list(gpu_info.keys())

        # Filter GPUs based on memory requirement
        gpus = [gpu for gpu in gpus if gpu_info[gpu]["free_memory"] >= required_memory]

        # Filter GPUs based on process count
        if exclusive:
            gpus = [gpu for gpu in gpus if gpu_info[gpu]["process_count"] == 0]

        # Sort GPUs by utilization and free memory
        if sort:
            gpus.sort(
                key=lambda x: (gpu_info[x]["utilization"], -gpu_info[x]["free_memory"])
            )

        # Fail if not enough GPUs are found
        if len(gpus) < num_gpus:
            return None

        return sorted(gpus[:num_gpus])

    def wait_for_gpu(
        self,
        required_memory: int,
        num_gpus: int,
        check_interval: float = 1.0,
        exclusive: bool = False,
        sort: bool = True,
    ) -> Tuple[Dict[int, Dict[str, Any]], List[int]]:
        """Wait for available GPUs and return their detailed info.

        Args:
            required_memory: Required memory in bytes for each GPU
            num_gpus: Number of GPUs needed
            check_interval: Time interval in seconds between checks
            exclusive: Whether to only return GPUs with no running processes
            sort: Whether to sort the GPUs by utilization and free memory

        Returns:
            Tuple of (Dictionary of GPU indices and their detailed info, List of selected GPU indices)
            Example: ({0: {'free_memory': ..., 'utilization': ...}, ...}, [0, 1])
        """

        while True:
            current_gpu_info = self._get_gpu_info()

            gpu_indices = self._find_available_gpu(
                gpu_info=current_gpu_info,
                required_memory=required_memory,
                num_gpus=num_gpus,
                sort=sort,
                exclusive=exclusive,
            )

            if gpu_indices is not None:
                return current_gpu_info, gpu_indices

            time.sleep(check_interval)
