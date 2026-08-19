import threading
import time
from dataclasses import dataclass

import psutil


@dataclass
class ResourceUsage:
    peak_cpu_percent: float
    peak_memory_mb: float
    elapsed_seconds: float


class ResourceMonitor:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.process = psutil.Process()
        self.cpu_count = psutil.cpu_count(logical=True) or 1
        self.stop_event = threading.Event()

    def start(self):
        self.peak_cpu = 0.0
        self.peak_memory = self.process.memory_info().rss
        self.start_time = time.perf_counter()
        self.stop_event.clear()

        # The first call has no meaningful statistical value and is used only for initialization
        self.process.cpu_percent(None)

        self.thread = threading.Thread(
            target=self._monitor,
            daemon=True
        )
        self.thread.start()

    def _sample(self):
        cpu = self.process.cpu_percent(None) / self.cpu_count
        memory = self.process.memory_info().rss

        self.peak_cpu = max(self.peak_cpu, cpu)
        self.peak_memory = max(self.peak_memory, memory)

    def _monitor(self):
        while not self.stop_event.wait(self.interval):
            self._sample()

    def stop(self):
        self.stop_event.set()
        self.thread.join()

        # Perform one final sampling to minimize the risk of missing any peak values before the process ends.
        self._sample()

        return ResourceUsage(
            peak_cpu_percent=self.peak_cpu,
            peak_memory_mb=self.peak_memory / 1024 / 1024,
            elapsed_seconds=time.perf_counter() - self.start_time,
        )


if __name__ == "__main__":
    # Example:
    # monitor = ResourceMonitor()
    
    # monitor.start()
    
    # try:
    #     r = my_func()  # Place the function to be monitored here
    # finally:
    #     usage = monitor.stop()
    
    # print(f"run time: {usage.elapsed_seconds:.2f}s")
    # print(f"Peak CPU: {usage.peak_cpu_percent:.1f}%")
    # print(f"Peak memory: {usage.peak_memory_mb:.1f} MB")

    pass
