import yaml
import argparse
import math
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Tuple

from torchinfo import summary

from models.utils import Model, parse_module


@dataclass
class CPUConfig:
    """CPU配置信息"""
    cores: int                    # CPU核心数
    base_freq: float         # 基础频率(GHz)
    max_freq: float        # 最大睿频(GHz)
    l3_cache: int             # L3缓存大小(MB)


@dataclass
class GPUConfig:
    """GPU配置信息"""
    num_gpus: int                # GPU数量
    vram: int                 # 显存大小(GB，单个GPU)
    vram_bandwidth: float # 显存带宽(GB/s，单个GPU)
    fp32: float       # FP32算力(TFLOPS，单个GPU)
    pcie_gen: int
    pcie_lanes: int
    nvlink_version: int


@dataclass
class RAMConfig:
    num_rams: int            # 内存通道数量
    ram_freq: int            # 内存频率(MHz)


@dataclass
class StorageConfig:
    """存储配置信息"""
    ssd_read: float        # SSD读取速度(GB/s)
    ssd_write: float       # SSD写入速度(GB/s)


@dataclass
class ModelConfig:
    """模型配置信息"""
    model_name: str             # 模型名称
    batch_size: int
    image_size: int
    total_params: int           # 总参数量
    trainable_params: int       # 可训练参数量
    tflops_per_sample: float = 0  # 每样本TFLOPS 1e12
    memory_per_sample: float = 0  # 每样本内存占用(MB)
    act_memory_per_sample: float = 0  # MB.


class HardwarePerformanceCalculator:
    """硬件性能计算器"""
    
    def __init__(self, cpu_config: CPUConfig, ram_config: RAMConfig, gpu_config: GPUConfig, storage_config: StorageConfig):
        self.cpu = cpu_config
        self.ram = ram_config
        self.gpu = gpu_config
        self.storage = storage_config

    def get_cpu_core_gflops(self) -> float:
        return self.cpu.max_freq * 2

    def get_ram_bandwidth(self) -> float:
        ram_data_width = 64  # 64-bit data width for both DDR4 and DDR5 RAM.
        ram_bandwidth = self.ram.num_rams * self.ram.ram_freq * ram_data_width / 8 / 1000  # GB/s.
        return ram_bandwidth
    
    def get_pcie_bandwidth(self) -> float:
        lane_bandwidth = {
            3: 0.80,    # PCIe 3.0: 0.8 GB/s per lane (8b/10b encoding: 8 GT/s * 0.8 bit / 8)
            4: 1.96,    # PCIe 4.0: 1.96 GB/s per lane (128b/130b encoding: 16 GT/s * 128/130 bit / 8)
            5: 3.94     # PCIe 5.0: 3.94 GB/s per lane (128b/130b encoding: 32 GT/s * 128/130 bit / 8)
        }
        gpu_pcie_bandwidth = lane_bandwidth.get(self.gpu.pcie_gen) * self.gpu.pcie_lanes
        return gpu_pcie_bandwidth

    def get_nvlink_bandwidth(self) -> float:
        nvlink_bandwidth = {
            1: 160,
            2: 300,
            3: 600,
            4: 900
        }  # GB/s.
        return nvlink_bandwidth.get(self.gpu.nvlink_version)


class ModelAnalyzer:
    
    @staticmethod
    def get_full_model_config(model_config_path: str, batch_size: int, image_size: int) -> ModelConfig:
        with open(model_config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        
        # Create model and get params / trainable params.
        model = Model(cfg)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Get FLOPS and memory cost.
        model_stats = summary(model, input_size=[1, 3, image_size, image_size], verbose=0)
        tflops = model_stats.total_mult_adds * 2 / 1e12  # MACs to FLOPs.
        memory_per_sample = 3 * (image_size ** 2) * 4 / (1024**2)  # FP32 memory cost, MB.

        # Activations memory cost. 100 MB for a 256 * 256 image.
        activation_memory_per_sample = 100 / (256 ** 2) * (image_size ** 2)
        
        return ModelConfig(
            model_name=model_config_path.split('/')[-1].replace('.yaml', ''),
            batch_size=batch_size,
            image_size=image_size,
            total_params=total_params,
            trainable_params=trainable_params,
            tflops_per_sample=tflops,
            memory_per_sample=memory_per_sample,
            act_memory_per_sample=activation_memory_per_sample,
        )
    
    @staticmethod
    def get_theoretical_batch_size(gpu_config: GPUConfig, model_config: ModelConfig) -> int:
        # Model parameters memory cost.
        param_memory_mb = model_config.trainable_params * 4 / (1024**2)  # FP32 parameters.
        optimizer_memory_mb = param_memory_mb * 2  # TODO: Adam optimizer.

        # Max batch size.
        available_memory_mb = gpu_config.vram * 1024 * 0.9  # Leave 10% VRAM.
        memory_for_batch_mb = available_memory_mb - param_memory_mb - optimizer_memory_mb
        max_batch_size = int(memory_for_batch_mb / (model_config.memory_per_sample + model_config.act_memory_per_sample))
        optimal_batch_size = 1
        while optimal_batch_size * 2 <= max_batch_size:
            optimal_batch_size *= 2
        
        return max(1, optimal_batch_size)


class TrainingTimeCalculator:
    """训练时间计算器 - 核心计算逻辑"""
    
    def __init__(self, hardware_calc: HardwarePerformanceCalculator):
        self.hw_calc = hardware_calc
    
    def single_gpu_training_time(self, model_config: ModelConfig) -> float:
        # Forward propagation.
        batch_size_per_gpu = int(model_config.batch_size / self.hw_calc.gpu.num_gpus)
        forward_tflops_per_batch = model_config.tflops_per_sample * batch_size_per_gpu
        gpu_tflops = self.hw_calc.gpu.fp32
        forward_time_per_batch = forward_tflops_per_batch / gpu_tflops  # Seconds.
        
        # Backward propagation (2-3x time cost compared to forward propagation.)
        backward_time_per_batch = forward_time_per_batch * 2.5  # Seconds.
        gpu_compute_time_per_batch = forward_time_per_batch + backward_time_per_batch
        return gpu_compute_time_per_batch

    
    def ddp_sync_time(self, model_config: ModelConfig, sync_mode: str) -> float:
        # Gradient data size.
        gradient_data_gb = model_config.trainable_params * 4 / (1024 ** 3)

        # Different sync mode.
        if sync_mode == "NVLINK" and self.hw_calc.gpu.nvlink_version is not None:
            nvlink_bw_gbps =  self.hw_calc.get_nvlink_bandwidth()
            # Ring AllReduce with NVLink: 2 * (N-1) / N * data_size per GPU
            sync_data_gb = 2 * (self.hw_calc.gpu.num_gpus - 1) / self.hw_calc.gpu.num_gpus * gradient_data_gb
            sync_time_ms = sync_data_gb / nvlink_bw_gbps * 1000
            return sync_time_ms
        elif sync_mode == "P2P":
            # DMA directly exchange data between GPU memory through PCIe.
            pcie_bw_gbps =  self.hw_calc.get_pcie_bandwidth()

            # Ring AllReduce:
            # Reduce: (N - 1) steps, send 1 / N and receive 1 / N.
            # Scatter-Reduce: (N - 1) steps, send 1 / N and receive 1 / N.
            # Since they are connected as a ring, the current GPU send is the next GPU receive.
            # 2 * (N-1) / N * data_size per GPU.
            sync_data_gb = 2 * (self.hw_calc.gpu.num_gpus - 1) / self.hw_calc.gpu.num_gpus * gradient_data_gb
            sync_time_ms = sync_data_gb / pcie_bw_gbps * 1000
            return sync_time_ms
        elif sync_mode == "RAM":
            # Scatter by RAM, not a ring.
            pcie_bw_gbps =  self.hw_calc.get_pcie_bandwidth()
            sync_data_gb = 2 * (self.hw_calc.gpu.num_gpus - 1) / self.hw_calc.gpu.num_gpus * gradient_data_gb * 2
            sync_pcie_time_ms = sync_data_gb / pcie_bw_gbps * 1000
            sync_ram_time_ms = sync_data_gb / self.hw_calc.get_ram_bandwidth() * 1000
            return max(sync_pcie_time_ms, sync_ram_time_ms)
        else:
            raise ValueError(f"Unsupported sync_mode: {sync_mode}")



def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deep Learning Model Training Time Calculator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Config file.
    parser.add_argument('--config', type=str, default='server_config.yaml')
    
    # Model config.
    parser.add_argument('--model-name', type=str, default='resnet50',
                       choices=['resnet50', 'resnet101', 'vgg16', 'vgg19'])
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--image-size', type=int, default=224)

    # CPU config.
    parser.add_argument('--cpu-cores', type=int, default=16)
    parser.add_argument('--cpu-base-freq', type=float, default=3.0)
    parser.add_argument('--cpu-max-freq', type=float, default=4.0)
    parser.add_argument('--cpu-l3-cache', type=int, default=32)

    # RAM config.
    parser.add_argument('--num-rams', type=int, default=4)
    parser.add_argument('--ram-freq', type=int, default=4800)
    
    # GPU config.
    parser.add_argument('--num-gpus', type=int, default=1)
    parser.add_argument('--gpu-vram', type=int, default=24)
    parser.add_argument('--gpu-vram-bw', type=float, default=1008.0)
    parser.add_argument('--gpu-fp32', type=float, default=82.6)
    parser.add_argument('--gpu-pcie-gen', type=int, default=4, choices=[3, 4, 5])
    parser.add_argument('--gpu-pcie-lanes', type=int, default=16)
    parser.add_argument('--nvlink-version', type=str, default=None)
    
    # Storage config.
    parser.add_argument('--ssd-read', type=float, default=7.0)
    parser.add_argument('--ssd-write', type=float, default=6.5)
    
    # Multi GPU sync mode.
    parser.add_argument('--sync-mode', type=str, default='P2P', choices=['P2P', 'NVLink', 'RAM'])
    
    # Output.
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--verbose', action='store_true')
    
    return parser.parse_args()


def load_config_from_yaml(config_path: str) -> Dict:
    """从YAML文件加载配置"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"YAML文件格式错误: {e}")


def merge_config_with_args(config: Dict, args: argparse.Namespace) -> argparse.Namespace:
    # Model config.
    if 'model' in config:
        model_config = config['model']
        if 'model_name' in model_config:
            args.model_name = model_config['model_name']
        if 'batch_size' in model_config:
            args.batch_size = model_config['batch_size']
        if 'image_size' in model_config:
            args.image_size = model_config['image_size']
    
    # CPU config.
    if 'cpu' in config:
        cpu_config = config['cpu']
        if 'cores' in cpu_config:
            args.cpu_cores = cpu_config['cores']
        if 'base_freq' in cpu_config:
            args.cpu_base_freq = cpu_config['base_freq']
        if 'max_freq' in cpu_config:
            args.cpu_max_freq = cpu_config['max_freq']
        if 'l3_cache' in cpu_config:
            args.cpu_l3_cache = cpu_config['l3_cache']

    # RAM config.
    if 'ram' in config:
        ram_config = config['ram']
        if 'num_rams' in ram_config:
            args.num_rams = ram_config['num_rams']
        if 'ram_freq' in ram_config:
            args.ram_freq = ram_config['ram_freq']
    
    # GPU config.
    if 'gpu' in config:
        gpu_config = config['gpu']
        if 'num_gpus' in gpu_config:
            args.num_gpus = gpu_config['num_gpus']
        if 'vram' in gpu_config:
            args.gpu_vram = gpu_config['vram']
        if 'vram_bw' in gpu_config:
            args.gpu_vram_bw = gpu_config['vram_bw']
        if 'fp32' in gpu_config:
            args.gpu_fp32 = gpu_config['fp32']
        if 'pcie_gen' in gpu_config:
            args.gpu_pcie_gen = gpu_config['pcie_gen']
        if 'pcie_lanes' in gpu_config:
            args.gpu_pcie_lanes = gpu_config['pcie_lanes']
        if 'nvlink_version' in gpu_config:
            args.nvlink_version = gpu_config['nvlink_version']
    
    # Storage config.
    if 'storage' in config:
        storage_config = config['storage']
        if 'ssd_read' in storage_config:
            args.ssd_read = storage_config['ssd_read']
        if 'ssd_write' in storage_config:
            args.ssd_write = storage_config['ssd_write']
    
    # Multi GPU sync mode.
    if 'sync_mode' in config:
        args.sync_mode = config['sync_mode']
    
    # Output.
    if 'output' in config:
        output_config = config['output']
        if 'report_file' in output_config and output_config['report_file']:
            args.output = output_config['report_file']
        if 'verbose' in output_config:
            args.verbose = output_config['verbose']
    
    return args


def create_hardware_configs(args: argparse.Namespace) -> Tuple[CPUConfig, RAMConfig, GPUConfig, StorageConfig]:
    cpu_config = CPUConfig(
        cores=args.cpu_cores,
        base_freq=args.cpu_base_freq,
        max_freq=args.cpu_max_freq,
        l3_cache=args.cpu_l3_cache,
    )

    ram_config = RAMConfig(
        num_rams=args.num_rams,
        ram_freq=args.ram_freq,
    )
    
    gpu_config = GPUConfig(
        num_gpus=args.num_gpus,
        vram=args.gpu_vram,
        vram_bandwidth=args.gpu_vram_bw,
        fp32=args.gpu_fp32,
        pcie_gen=args.gpu_pcie_gen,
        pcie_lanes=args.gpu_pcie_lanes,
        nvlink_version=args.nvlink_version,
    )
    
    storage_config = StorageConfig(
        ssd_read=args.ssd_read,
        ssd_write=args.ssd_write
    )
    
    return cpu_config, ram_config, gpu_config, storage_config


def main():
    """主函数"""
    print("=== 深度学习模型训练时间理论计算器 ===")
    
    # 解析命令行参数
    args = parse_arguments()
    if args.config:
        try:
            print(f"正在加载配置文件: {args.config}")
            config = load_config_from_yaml(args.config)
            args = merge_config_with_args(config, args)
            print("配置文件加载成功")
        except (FileNotFoundError, ValueError) as e:
            print(f"配置文件加载失败: {e}")
            return
    
    if args.verbose:
        print(f"模型: {args.model_name}")
        print(f"批次大小: {args.batch_size}")
        print(f"GPU数量: {args.num_gpus}")
        print(f"同步模式: {args.sync_mode}")
    
    try:
        cpu_config, ram_config, gpu_config, storage_config = create_hardware_configs(args)
        hw_calc = HardwarePerformanceCalculator(cpu_config, ram_config, gpu_config, storage_config)

        # Model config.
        model_config_path = f"configs/{args.model_name}.yaml"
        model_config = ModelAnalyzer.get_full_model_config(model_config_path, args.batch_size, args.image_size)

        # Optimize batch size.
        batch_size_per_gpu = ModelAnalyzer.get_theoretical_batch_size(gpu_config, model_config)
        if args.batch_size != batch_size_per_gpu * gpu_config.num_gpus and args.verbose:
            print(f"建议优化批次大小: {batch_size_per_gpu} (当前: {args.batch_size})")

        # We consider a single batch to calculate all time cost during i-th iteration of multi-gpu training.
        # Data loading time.
        data_gb = model_config.batch_size * model_config.memory_per_sample / 1024
        data_load_time_ms = data_gb / storage_config.ssd_read * 1000
        print(f"Load batch data from SSD: {data_load_time_ms} ms.")

        # Pre-processing data for i+1-th iteration.
        # CPU - RAM operations during pre-processing:
        # 1. Dtype: from uint8 to float32. (RAM -> CPU -> RAM)
        # 2. Layout: [H, W, C] -> [C, H, W]. (RAM -> CPU -> RAM)
        # 3. Normalize: to 0-1. (RAM -> CPU -> RAM) ~4 FLOPS / pixel
        # 4. Mosaic: read 4 images, write 1.
        # 5. Affine/Perspective: read 1, write 1.  ~10 FLOPS / pixel
        # 6. Flip: read 1, write 1.
        # 7. Color aug: read 1, write 1.
        image_mb = 3 * (model_config.image_size ** 2) * 4 / (1024 ** 2)
        image_preprocess_mb = 20 * image_mb  # Approximated to 20 CPU-to-RAM reading and writing.
        batch_preprocess_mb = model_config.batch_size * image_preprocess_mb
        ram_bw = hw_calc.get_ram_bandwidth()
        ram_time_ms = batch_preprocess_mb / 1024 / ram_bw * 1000
        print(f"RAM time when CPU pre-processing batch data: {ram_time_ms} ms.")

        # CPU computing time, approximated 20 FLOPS per pixel. Using 8 cores CPU per GPU.
        cpu_cores_gflops = gpu_config.num_gpus * 8 * hw_calc.get_cpu_core_gflops()
        image_preprocess_gflops = 20 * (model_config.batch_size * 3 * model_config.image_size ** 2) / 1e9
        cpu_time_ms = image_preprocess_gflops / cpu_cores_gflops * 1000
        print(f"CPU pre-processing batch data: {cpu_time_ms} ms.")

        # RAM to GPU transferring time (n GPU in parallel).
        data_per_gpu_gb = model_config.batch_size / gpu_config.num_gpus * image_mb / 1024
        pcie_bw = hw_calc.get_pcie_bandwidth()
        pcie_time_ms = data_per_gpu_gb / pcie_bw * 1000
        print(f"PCIe data transfer time from RAM to GPU: {pcie_time_ms} ms.")

        # GPU computing time.
        training_calc = TrainingTimeCalculator(hw_calc)
        gpu_compute_time_ms = training_calc.single_gpu_training_time(model_config) * 1000
        print(f"GPU computing time of forward and backward propagation: {gpu_compute_time_ms} ms.")

        # Multi-GPU gradient sync time.
        sync_time = training_calc.ddp_sync_time(model_config, args.sync_mode)
        print(f"Gradient data sync time when using multi GPU DDP training: {sync_time} ms.")
        
    except FileNotFoundError:
        model_config_path = f"configs/{args.model}.yaml"
        print(f"错误: 找不到模型配置文件 {model_config_path}")
        print("请确保模型配置文件存在于 configs/ 目录中")
    except Exception as e:
        print(f"计算过程中发生错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()