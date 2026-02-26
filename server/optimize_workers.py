#!/usr/bin/env python3
"""
Worker Optimization Script for Butterflyfishes Classification Training

This script optimizes the num_workers parameter for data loading by:
1. Running training with different num_workers values
2. Monitoring system resources (CPU, GPU, Memory, I/O)
3. Measuring training throughput and latency
4. Finding the optimal num_workers setting

Usage:
    python optimize_workers.py --cfg ./configs/butterflyfishes.yaml --max-workers 16 --epochs-per-test 2
"""

import argparse
import time
import psutil
import GPUtil
import threading
import yaml
from pathlib import Path
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
import sys
import os

# Add parent directory to path to import training script
sys.path.append(str(Path(__file__).parent.parent))
from fam_cls_train import parse_args, train as train_main


@dataclass
class SystemMetrics:
    """System resource metrics during training"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    gpu_utilization: float
    gpu_memory_percent: float
    disk_io_read: float
    disk_io_write: float
    network_io_sent: float
    network_io_recv: float


@dataclass
class TrainingMetrics:
    """Training performance metrics"""
    num_workers: int
    total_time: float
    avg_iteration_time: float
    throughput_samples_per_sec: float
    max_cpu_percent: float
    max_memory_percent: float
    max_gpu_utilization: float
    max_gpu_memory_percent: float
    avg_cpu_percent: float
    avg_memory_percent: float
    avg_gpu_utilization: float
    avg_gpu_memory_percent: float


class SystemMonitor:
    """Monitor system resources during training"""
    
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.metrics: List[SystemMetrics] = []
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start monitoring in a separate thread"""
        self.monitoring = True
        self.metrics = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
            
    def _monitor_loop(self):
        """Main monitoring loop"""
        # Get initial disk and network I/O
        disk_io = psutil.disk_io_counters()
        net_io = psutil.net_io_counters()
        last_disk_read = disk_io.read_bytes if disk_io else 0
        last_disk_write = disk_io.write_bytes if disk_io else 0
        last_net_sent = net_io.bytes_sent if net_io else 0
        last_net_recv = net_io.bytes_recv if net_io else 0
        last_time = time.time()
        
        while self.monitoring:
            try:
                # CPU and Memory
                cpu_percent = psutil.cpu_percent(interval=None)
                memory_percent = psutil.virtual_memory().percent
                
                # GPU metrics
                gpu_utilization = 0.0
                gpu_memory_percent = 0.0
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu_utilization = gpus[0].load * 100
                        gpu_memory_percent = (gpus[0].memoryUsed / gpus[0].memoryTotal) * 100
                except:
                    pass
                
                # Disk and Network I/O
                current_time = time.time()
                time_diff = current_time - last_time
                
                disk_io = psutil.disk_io_counters()
                net_io = psutil.net_io_counters()
                
                if disk_io and net_io:
                    disk_read_rate = (disk_io.read_bytes - last_disk_read) / time_diff / 1024 / 1024  # MB/s
                    disk_write_rate = (disk_io.write_bytes - last_disk_write) / time_diff / 1024 / 1024  # MB/s
                    net_sent_rate = (net_io.bytes_sent - last_net_sent) / time_diff / 1024 / 1024  # MB/s
                    net_recv_rate = (net_io.bytes_recv - last_net_recv) / time_diff / 1024 / 1024  # MB/s
                    
                    last_disk_read = disk_io.read_bytes
                    last_disk_write = disk_io.write_bytes
                    last_net_sent = net_io.bytes_sent
                    last_net_recv = net_io.bytes_recv
                else:
                    disk_read_rate = disk_write_rate = net_sent_rate = net_recv_rate = 0.0
                
                last_time = current_time
                
                # Store metrics
                metric = SystemMetrics(
                    timestamp=current_time,
                    cpu_percent=cpu_percent,
                    memory_percent=memory_percent,
                    gpu_utilization=gpu_utilization,
                    gpu_memory_percent=gpu_memory_percent,
                    disk_io_read=disk_read_rate,
                    disk_io_write=disk_write_rate,
                    network_io_sent=net_sent_rate,
                    network_io_recv=net_recv_rate
                )
                self.metrics.append(metric)
                
                time.sleep(self.interval)
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(self.interval)
    
    def get_summary(self) -> Dict:
        """Get summary statistics of collected metrics"""
        if not self.metrics:
            return {}
            
        cpu_percents = [m.cpu_percent for m in self.metrics]
        memory_percents = [m.memory_percent for m in self.metrics]
        gpu_utils = [m.gpu_utilization for m in self.metrics]
        gpu_memory_percents = [m.gpu_memory_percent for m in self.metrics]
        
        return {
            'max_cpu_percent': max(cpu_percents),
            'avg_cpu_percent': sum(cpu_percents) / len(cpu_percents),
            'max_memory_percent': max(memory_percents),
            'avg_memory_percent': sum(memory_percents) / len(memory_percents),
            'max_gpu_utilization': max(gpu_utils),
            'avg_gpu_utilization': sum(gpu_utils) / len(gpu_utils),
            'max_gpu_memory_percent': max(gpu_memory_percents),
            'avg_gpu_memory_percent': sum(gpu_memory_percents) / len(gpu_memory_percents),
            'monitoring_duration': self.metrics[-1].timestamp - self.metrics[0].timestamp
        }


class WorkerOptimizer:
    """Optimize num_workers parameter for data loading"""
    
    def __init__(self, config_path: str, max_workers: int = 16, epochs_per_test: int = 2):
        self.config_path = config_path
        self.max_workers = max_workers
        self.epochs_per_test = epochs_per_test
        self.results: List[TrainingMetrics] = []
        self.monitor = SystemMonitor()
        
    def run_training_with_workers(self, num_workers: int) -> TrainingMetrics:
        """Run training with specific num_workers and collect metrics"""
        print(f"\n{'='*60}")
        print(f"Testing with num_workers = {num_workers}")
        print(f"{'='*60}")
        
        # Modify config to set num_workers
        self._update_config_num_workers(num_workers)
        
        # Start monitoring
        self.monitor.start_monitoring()
        start_time = time.time()
        
        try:
            # Run training
            args = parse_args(["--cfg", self.config_path])
            args.epochs = self.epochs_per_test
            args.validate_epoch = None  # Disable validation for faster testing
            
            # Redirect stdout to capture training output
            original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            
            train_main(args)
            
            sys.stdout.close()
            sys.stdout = original_stdout
            
        except Exception as e:
            print(f"Training error with num_workers={num_workers}: {e}")
            self.monitor.stop_monitoring()
            return None
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        summary = self.monitor.get_summary()
        
        # Estimate throughput (samples per second)
        # This is approximate since we don't have exact sample count
        estimated_samples_per_epoch = 1000  # Adjust based on your dataset
        total_samples = estimated_samples_per_epoch * self.epochs_per_test
        throughput = total_samples / total_time
        
        metrics = TrainingMetrics(
            num_workers=num_workers,
            total_time=total_time,
            avg_iteration_time=total_time / (self.epochs_per_test * 100),  # Approximate iterations
            throughput_samples_per_sec=throughput,
            max_cpu_percent=summary.get('max_cpu_percent', 0),
            max_memory_percent=summary.get('max_memory_percent', 0),
            max_gpu_utilization=summary.get('max_gpu_utilization', 0),
            max_gpu_memory_percent=summary.get('max_gpu_memory_percent', 0),
            avg_cpu_percent=summary.get('avg_cpu_percent', 0),
            avg_memory_percent=summary.get('avg_memory_percent', 0),
            avg_gpu_utilization=summary.get('avg_gpu_utilization', 0),
            avg_gpu_memory_percent=summary.get('avg_gpu_memory_percent', 0)
        )
        
        print(f"Results for num_workers={num_workers}:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {throughput:.2f} samples/sec")
        print(f"  Max CPU: {metrics.max_cpu_percent:.1f}%")
        print(f"  Max Memory: {metrics.max_memory_percent:.1f}%")
        print(f"  Max GPU: {metrics.max_gpu_utilization:.1f}%")
        
        return metrics
    
    def _update_config_num_workers(self, num_workers: int):
        """Update the config file with new num_workers value"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update num_workers in the config
        if 'dataloader' in config:
            config['dataloader']['num_workers'] = num_workers
        else:
            config['dataloader'] = {'num_workers': num_workers}
        
        # Write back to config
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def optimize(self) -> TrainingMetrics:
        """Run optimization across different num_workers values"""
        print("Starting num_workers optimization...")
        print(f"Testing range: 0 to {self.max_workers}")
        print(f"Epochs per test: {self.epochs_per_test}")
        
        # Test different num_workers values
        test_values = [0, 1, 2, 4, 8, 12, 16]
        test_values = [v for v in test_values if v <= self.max_workers]
        
        for num_workers in test_values:
            metrics = self.run_training_with_workers(num_workers)
            if metrics:
                self.results.append(metrics)
            
            # Small break between tests
            time.sleep(2)
        
        # Find optimal setting
        if self.results:
            optimal = self._find_optimal_setting()
            self._save_results()
            self._plot_results()
            return optimal
        
        return None
    
    def _find_optimal_setting(self) -> TrainingMetrics:
        """Find the optimal num_workers setting based on throughput and resource usage"""
        if not self.results:
            return None
        
        # Score each setting based on throughput and resource efficiency
        scored_results = []
        for result in self.results:
            # Normalize metrics (0-1 scale)
            throughput_score = result.throughput_samples_per_sec / max(r.throughput_samples_per_sec for r in self.results)
            cpu_efficiency = 1 - (result.avg_cpu_percent / 100)  # Lower CPU usage is better
            memory_efficiency = 1 - (result.avg_memory_percent / 100)  # Lower memory usage is better
            
            # Combined score (weighted)
            score = (0.6 * throughput_score + 
                    0.2 * cpu_efficiency + 
                    0.2 * memory_efficiency)
            
            scored_results.append((result, score))
        
        # Sort by score and return the best
        scored_results.sort(key=lambda x: x[1], reverse=True)
        optimal = scored_results[0][0]
        
        print(f"\n{'='*60}")
        print(f"OPTIMAL SETTING: num_workers = {optimal.num_workers}")
        print(f"Score: {scored_results[0][1]:.3f}")
        print(f"Throughput: {optimal.throughput_samples_per_sec:.2f} samples/sec")
        print(f"Avg CPU: {optimal.avg_cpu_percent:.1f}%")
        print(f"Avg Memory: {optimal.avg_memory_percent:.1f}%")
        print(f"{'='*60}")
        
        return optimal
    
    def _save_results(self):
        """Save optimization results to CSV"""
        if not self.results:
            return
        
        data = []
        for result in self.results:
            data.append({
                'num_workers': result.num_workers,
                'total_time': result.total_time,
                'throughput_samples_per_sec': result.throughput_samples_per_sec,
                'avg_iteration_time': result.avg_iteration_time,
                'max_cpu_percent': result.max_cpu_percent,
                'avg_cpu_percent': result.avg_cpu_percent,
                'max_memory_percent': result.max_memory_percent,
                'avg_memory_percent': result.avg_memory_percent,
                'max_gpu_utilization': result.max_gpu_utilization,
                'avg_gpu_utilization': result.avg_gpu_utilization,
                'max_gpu_memory_percent': result.max_gpu_memory_percent,
                'avg_gpu_memory_percent': result.avg_gpu_memory_percent
            })
        
        df = pd.DataFrame(data)
        output_file = Path(__file__).parent / "worker_optimization_results.csv"
        df.to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")
    
    def _plot_results(self):
        """Create visualization of optimization results"""
        if not self.results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('num_workers Optimization Results', fontsize=16)
        
        workers = [r.num_workers for r in self.results]
        throughput = [r.throughput_samples_per_sec for r in self.results]
        cpu_avg = [r.avg_cpu_percent for r in self.results]
        memory_avg = [r.avg_memory_percent for r in self.results]
        gpu_avg = [r.avg_gpu_utilization for r in self.results]
        
        # Throughput
        axes[0, 0].plot(workers, throughput, 'bo-', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('num_workers')
        axes[0, 0].set_ylabel('Throughput (samples/sec)')
        axes[0, 0].set_title('Training Throughput')
        axes[0, 0].grid(True)
        
        # CPU Usage
        axes[0, 1].plot(workers, cpu_avg, 'ro-', linewidth=2, markersize=8)
        axes[0, 1].set_xlabel('num_workers')
        axes[0, 1].set_ylabel('CPU Usage (%)')
        axes[0, 1].set_title('Average CPU Usage')
        axes[0, 1].grid(True)
        
        # Memory Usage
        axes[1, 0].plot(workers, memory_avg, 'go-', linewidth=2, markersize=8)
        axes[1, 0].set_xlabel('num_workers')
        axes[1, 0].set_ylabel('Memory Usage (%)')
        axes[1, 0].set_title('Average Memory Usage')
        axes[1, 0].grid(True)
        
        # GPU Usage
        axes[1, 1].plot(workers, gpu_avg, 'mo-', linewidth=2, markersize=8)
        axes[1, 1].set_xlabel('num_workers')
        axes[1, 1].set_ylabel('GPU Utilization (%)')
        axes[1, 1].set_title('Average GPU Utilization')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        output_file = Path(__file__).parent / "worker_optimization_plots.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plots saved to: {output_file}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Optimize num_workers for data loading')
    parser.add_argument('--cfg', type=str, required=True, 
                       help='Path to training config file')
    parser.add_argument('--max-workers', type=int, default=16,
                       help='Maximum number of workers to test')
    parser.add_argument('--epochs-per-test', type=int, default=2,
                       help='Number of epochs to run for each worker test')
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not Path(args.cfg).exists():
        print(f"Error: Config file {args.cfg} not found!")
        return
    
    # Run optimization
    optimizer = WorkerOptimizer(
        config_path=args.cfg,
        max_workers=args.max_workers,
        epochs_per_test=args.epochs_per_test
    )
    
    optimal = optimizer.optimize()
    
    if optimal:
        print(f"\nOptimization completed!")
        print(f"Recommended num_workers: {optimal.num_workers}")
    else:
        print("\nOptimization failed!")


if __name__ == "__main__":
    main() 