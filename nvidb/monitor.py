"""
GPU Monitor Decorator for nvidb

Provides @nvidb.monitor decorator to track GPU usage statistics
during function execution, including duration, peak memory, and utilization.
"""

import functools
import inspect
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Dict, Any
import asyncio

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


@dataclass
class GPUSnapshot:
    """Single GPU state snapshot"""
    timestamp: float
    gpu_index: int
    memory_used: int  # bytes
    memory_total: int  # bytes
    gpu_utilization: int  # percentage
    memory_utilization: int  # percentage
    temperature: int  # celsius
    power_usage: float  # watts


@dataclass
class GPUStats:
    """Aggregated GPU statistics for a monitoring session"""
    gpu_index: int
    gpu_name: str
    memory_total: int  # bytes
    
    # Memory stats
    memory_start: int = 0
    memory_end: int = 0
    memory_peak: int = 0
    memory_min: int = 0
    
    # Utilization stats
    utilization_samples: List[int] = field(default_factory=list)
    
    # Temperature stats
    temperature_peak: int = 0
    temperature_samples: List[int] = field(default_factory=list)
    
    # Power stats
    power_peak: float = 0.0
    power_samples: List[float] = field(default_factory=list)
    
    @property
    def memory_delta(self) -> int:
        return self.memory_end - self.memory_start
    
    @property
    def avg_utilization(self) -> float:
        if not self.utilization_samples:
            return 0.0
        return sum(self.utilization_samples) / len(self.utilization_samples)
    
    @property
    def avg_temperature(self) -> float:
        if not self.temperature_samples:
            return 0.0
        return sum(self.temperature_samples) / len(self.temperature_samples)
    
    @property
    def avg_power(self) -> float:
        if not self.power_samples:
            return 0.0
        return sum(self.power_samples) / len(self.power_samples)


class GPUMonitor:
    """GPU monitoring context manager and sampler"""
    
    def __init__(self, sample_interval: float = 0.1, gpu_indices: Optional[List[int]] = None):
        """
        Initialize GPU monitor.
        
        Args:
            sample_interval: Sampling interval in seconds (default 100ms)
            gpu_indices: List of GPU indices to monitor. If None, monitors all available GPUs.
        """
        self.sample_interval = sample_interval
        self.gpu_indices = gpu_indices
        self._stop_event = threading.Event()
        self._sample_thread: Optional[threading.Thread] = None
        self._snapshots: Dict[int, List[GPUSnapshot]] = {}
        self._handles: Dict[int, Any] = {}
        self._gpu_names: Dict[int, str] = {}
        self._initialized = False
        self._lock = threading.Lock()
    
    def _init_nvml(self) -> bool:
        """Initialize NVML and get GPU handles"""
        if not PYNVML_AVAILABLE:
            return False
        
        try:
            pynvml.nvmlInit()
            gpu_count = pynvml.nvmlDeviceGetCount()
            
            if self.gpu_indices is None:
                self.gpu_indices = list(range(gpu_count))
            
            for idx in self.gpu_indices:
                if idx < gpu_count:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                    self._handles[idx] = handle
                    self._gpu_names[idx] = pynvml.nvmlDeviceGetName(handle)
                    self._snapshots[idx] = []
            
            self._initialized = True
            return True
        except Exception as e:
            print(f"[nvidb.monitor] Warning: Failed to initialize NVML: {e}")
            return False
    
    def _shutdown_nvml(self):
        """Shutdown NVML"""
        if self._initialized:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
            self._initialized = False
    
    def _take_snapshot(self) -> Dict[int, GPUSnapshot]:
        """Take a snapshot of all monitored GPUs"""
        snapshots = {}
        timestamp = time.time()
        
        for idx, handle in self._handles.items():
            try:
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                except:
                    power = 0.0
                
                snapshot = GPUSnapshot(
                    timestamp=timestamp,
                    gpu_index=idx,
                    memory_used=memory_info.used,
                    memory_total=memory_info.total,
                    gpu_utilization=utilization.gpu,
                    memory_utilization=utilization.memory,
                    temperature=temperature,
                    power_usage=power
                )
                snapshots[idx] = snapshot
                
                with self._lock:
                    self._snapshots[idx].append(snapshot)
                    
            except Exception as e:
                pass  # Skip failed snapshots
        
        return snapshots
    
    def _sampling_loop(self):
        """Background sampling loop"""
        while not self._stop_event.is_set():
            self._take_snapshot()
            self._stop_event.wait(self.sample_interval)
    
    def start(self):
        """Start GPU monitoring"""
        if not self._init_nvml():
            return False
        
        self._stop_event.clear()
        self._sample_thread = threading.Thread(target=self._sampling_loop, daemon=True)
        self._sample_thread.start()
        
        # Take initial snapshot
        self._take_snapshot()
        return True
    
    def stop(self) -> Dict[int, GPUStats]:
        """Stop monitoring and return aggregated stats"""
        # Take final snapshot
        self._take_snapshot()
        
        # Stop sampling thread
        self._stop_event.set()
        if self._sample_thread:
            self._sample_thread.join(timeout=1.0)
        
        # Aggregate stats
        stats = {}
        with self._lock:
            for idx, snapshots in self._snapshots.items():
                if not snapshots:
                    continue
                
                gpu_stats = GPUStats(
                    gpu_index=idx,
                    gpu_name=self._gpu_names.get(idx, f"GPU {idx}"),
                    memory_total=snapshots[0].memory_total if snapshots else 0
                )
                
                gpu_stats.memory_start = snapshots[0].memory_used
                gpu_stats.memory_end = snapshots[-1].memory_used
                gpu_stats.memory_peak = max(s.memory_used for s in snapshots)
                gpu_stats.memory_min = min(s.memory_used for s in snapshots)
                
                gpu_stats.utilization_samples = [s.gpu_utilization for s in snapshots]
                gpu_stats.temperature_samples = [s.temperature for s in snapshots]
                gpu_stats.temperature_peak = max(s.temperature for s in snapshots)
                gpu_stats.power_samples = [s.power_usage for s in snapshots]
                gpu_stats.power_peak = max(s.power_usage for s in snapshots)
                
                stats[idx] = gpu_stats
        
        self._shutdown_nvml()
        return stats


def _format_bytes(size_bytes: int) -> str:
    """Format bytes to human readable string"""
    if size_bytes >= 1024 ** 3:
        return f"{size_bytes / (1024 ** 3):.2f} GiB"
    elif size_bytes >= 1024 ** 2:
        return f"{size_bytes / (1024 ** 2):.2f} MiB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.2f} KiB"
    return f"{size_bytes} B"


def _format_delta(delta_bytes: int) -> str:
    """Format memory delta with sign"""
    sign = "+" if delta_bytes >= 0 else ""
    return f"{sign}{_format_bytes(abs(delta_bytes))}"


def _get_caller_info(func: Callable) -> tuple:
    """Get function signature, file path, and line number"""
    try:
        sig = inspect.signature(func)
        source_file = inspect.getfile(func)
        source_lines = inspect.getsourcelines(func)
        line_number = source_lines[1]
        return sig, source_file, line_number
    except Exception:
        return None, None, None


def _print_stats(
    func_name: str,
    duration: float,
    stats: Dict[int, GPUStats],
    signature: Optional[inspect.Signature],
    source_file: Optional[str],
    line_number: Optional[int]
):
    """Print formatted GPU stats to console"""
    print()
    print("=" * 70)
    print(f"[nvidb.monitor] Function completed: {func_name}")
    
    # Print function signature and location
    if signature:
        print(f"  Signature: {func_name}{signature}")
    if source_file and line_number:
        print(f"  Location: {source_file}:{line_number}")
    
    print("-" * 70)
    print(f"  Duration: {duration:.3f}s")
    print("-" * 70)
    
    if not stats:
        print("  No GPU data collected (NVML unavailable or no GPUs found)")
    else:
        for idx, gpu_stat in sorted(stats.items()):
            print(f"  GPU {idx}: {gpu_stat.gpu_name}")
            print(f"    Memory:")
            print(f"      Peak:    {_format_bytes(gpu_stat.memory_peak)} / {_format_bytes(gpu_stat.memory_total)}")
            print(f"      Delta:   {_format_delta(gpu_stat.memory_delta)}")
            print(f"      Start:   {_format_bytes(gpu_stat.memory_start)}")
            print(f"      End:     {_format_bytes(gpu_stat.memory_end)}")
            print(f"    Utilization:")
            print(f"      Avg:     {gpu_stat.avg_utilization:.1f}%")
            print(f"      Samples: {len(gpu_stat.utilization_samples)}")
            print(f"    Temperature:")
            print(f"      Peak:    {gpu_stat.temperature_peak}C")
            print(f"      Avg:     {gpu_stat.avg_temperature:.1f}C")
            if gpu_stat.power_samples and any(p > 0 for p in gpu_stat.power_samples):
                print(f"    Power:")
                print(f"      Peak:    {gpu_stat.power_peak:.1f}W")
                print(f"      Avg:     {gpu_stat.avg_power:.1f}W")
    
    print("=" * 70)
    print()


def monitor(
    func: Optional[Callable] = None,
    *,
    sample_interval: float = 0.1,
    gpu_indices: Optional[List[int]] = None,
    enabled: bool = True
):
    """
    Decorator to monitor GPU usage during function execution.
    
    Can be used as @monitor or @monitor(sample_interval=0.05)
    
    Args:
        func: The function to wrap (when used without parentheses)
        sample_interval: Sampling interval in seconds (default 100ms)
        gpu_indices: List of GPU indices to monitor. If None, monitors all GPUs.
        enabled: Whether monitoring is enabled. Set to False to disable.
    
    Usage:
        @nvidb.monitor
        def train():
            ...
        
        @nvidb.monitor(sample_interval=0.05, gpu_indices=[0, 1])
        def train_multi_gpu():
            ...
        
        @nvidb.monitor
        async def async_train():
            ...
    """
    
    def decorator(fn: Callable):
        # Get function info at decoration time
        signature, source_file, line_number = _get_caller_info(fn)
        
        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            if not enabled:
                return fn(*args, **kwargs)
            
            gpu_monitor = GPUMonitor(
                sample_interval=sample_interval,
                gpu_indices=gpu_indices
            )
            
            start_time = time.time()
            gpu_monitor.start()
            
            try:
                result = fn(*args, **kwargs)
                return result
            finally:
                stats = gpu_monitor.stop()
                duration = time.time() - start_time
                _print_stats(fn.__name__, duration, stats, signature, source_file, line_number)
        
        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            if not enabled:
                return await fn(*args, **kwargs)
            
            gpu_monitor = GPUMonitor(
                sample_interval=sample_interval,
                gpu_indices=gpu_indices
            )
            
            start_time = time.time()
            gpu_monitor.start()
            
            try:
                result = await fn(*args, **kwargs)
                return result
            finally:
                stats = gpu_monitor.stop()
                duration = time.time() - start_time
                _print_stats(fn.__name__, duration, stats, signature, source_file, line_number)
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(fn):
            return async_wrapper
        return sync_wrapper
    
    # Handle both @monitor and @monitor() usage
    if func is not None:
        return decorator(func)
    return decorator


# Convenience alias
gpu_monitor = monitor
