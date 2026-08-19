from __future__ import annotations

import logging
import platform
import psutil
import winreg
from dataclasses import dataclass
from typing import Optional

from pynvml import (
    NVMLError,
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetName,
    nvmlInit,
    nvmlShutdown,
)

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class SystemInfo:
    cpu_model: str
    ram_size_gb: int
    gpu_models: list[str]


def get_cpu_model() -> str:
    """Return the CPU model name."""
    try:
        cpu_model = platform.processor().strip()

        if cpu_model:
            return cpu_model

    except Exception:
        pass

    return "Unknown"


def get_ram_size_gb() -> int:
    """Return installed physical RAM rounded to the nearest GB."""
    try:
        total_bytes = psutil.virtual_memory().total
        return round(total_bytes / (1024 ** 3))
    except Exception:
        return 0


def get_nvidia_gpu_models() -> list[str]:
    """
    Return NVIDIA GPU model names using NVML.

    NVML initialization/shutdown is always handled safely.
    If NVIDIA drivers/NVML are unavailable, an empty list is returned.
    """
    gpus: list[str] = []
    initialized = False

    try:
        nvmlInit()
        initialized = True

        gpu_count = nvmlDeviceGetCount()

        for index in range(gpu_count):
            try:
                handle = nvmlDeviceGetHandleByIndex(index)
                name = nvmlDeviceGetName(handle)

                if isinstance(name, bytes):
                    name = name.decode("utf-8", errors="replace")

                name = name.strip()

                if name:
                    gpus.append(name)

            except NVMLError:
                # One GPU failing should not prevent other GPUs
                # from being detected.
                continue

    except NVMLError:
        # NVIDIA driver/NVML is unavailable.
        pass

    except Exception:
        # Do not allow hardware detection to break the application.
        pass

    finally:
        if initialized:
            try:
                nvmlShutdown()
            except Exception:
                pass

    return gpus


def get_windows_gpu_models() -> list[str]:
    """
    Detect display adapters from the Windows registry.

    This avoids PowerShell/WMI and is useful for Intel/AMD GPUs
    that are not visible through NVIDIA NVML.
    """
    gpus: list[str] = []

    registry_paths = (
        r"SYSTEM\CurrentControlSet\Control\Video",
    )

    try:
        for registry_path in registry_paths:
            try:
                with winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    registry_path,
                ) as video_key:

                    subkey_count = winreg.QueryInfoKey(video_key)[0]

                    for index in range(subkey_count):
                        try:
                            adapter_key_name = winreg.EnumKey(
                                video_key,
                                index,
                            )

                            adapter_path = (
                                f"{registry_path}\\{adapter_key_name}"
                            )

                            with winreg.OpenKey(
                                winreg.HKEY_LOCAL_MACHINE,
                                adapter_path,
                            ) as adapter_key:

                                subkey_count = winreg.QueryInfoKey(
                                    adapter_key
                                )[0]

                                for sub_index in range(subkey_count):
                                    try:
                                        subkey_name = winreg.EnumKey(
                                            adapter_key,
                                            sub_index,
                                        )

                                        subkey_path = (
                                            f"{adapter_path}\\"
                                            f"{subkey_name}"
                                        )

                                        with winreg.OpenKey(
                                            winreg.HKEY_LOCAL_MACHINE,
                                            subkey_path,
                                        ) as display_key:

                                            try:
                                                name, _ = winreg.QueryValueEx(
                                                    display_key,
                                                    "DriverDesc",
                                                )
                                            except FileNotFoundError:
                                                continue

                                            if (
                                                isinstance(name, str)
                                                and name.strip()
                                                and name.strip()
                                                not in gpus
                                            ):
                                                gpus.append(name.strip())

                                    except (
                                        OSError,
                                        PermissionError,
                                    ):
                                        continue

                        except (
                            OSError,
                            PermissionError,
                        ):
                            continue

            except (
                OSError,
                PermissionError,
            ):
                continue

    except Exception:
        pass

    return gpus


def get_gpu_models() -> list[str]:
    """
    Return available GPU models.

    NVIDIA GPUs are detected using NVML.
    Other display adapters are detected using the Windows registry.

    NVIDIA results are preferred to avoid duplicate NVIDIA entries.
    """
    nvidia_gpus = get_nvidia_gpu_models()
    windows_gpus = get_windows_gpu_models()

    result: list[str] = []

    # Add NVIDIA GPUs detected by NVML first.
    for gpu in nvidia_gpus:
        if gpu not in result:
            result.append(gpu)

    # Add GPUs from Windows registry.
    for gpu in windows_gpus:
        # Avoid adding NVIDIA GPU twice.
        if any(
            gpu.lower() in existing.lower()
            or existing.lower() in gpu.lower()
            for existing in result
        ):
            continue

        result.append(gpu)

    return result


def get_system_info() -> SystemInfo:
    """Collect CPU, RAM, and GPU information."""
    return SystemInfo(
        cpu_model=get_cpu_model(),
        ram_size_gb=get_ram_size_gb(),
        gpu_models=get_gpu_models(),
    )


def print_system_info(info: SystemInfo) -> None:
    """Print system information in the requested format."""
    
    gpu_text = "\n".join(
        f"GPU {i}        : {gpu}"
        for i, gpu in enumerate(info.gpu_models, start=1)
    )

    system_info = (
        f"\n----- System Information -----"
        f"\nCPU Model    : {info.cpu_model}"
        f"\nRAM Size     : {info.ram_size_gb} GB"
        f"\n{gpu_text}"
        f"\n------------------------------"
        f"\n"
    )
    logger.info(system_info)
    # print(system_info)


if __name__ == "__main__":
    system_info = get_system_info()
    print_system_info(system_info)