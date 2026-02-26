# Worker Optimization Script

这个脚本用于优化深度学习训练中的 `num_workers` 参数，通过监控系统资源使用情况和训练性能来自动找到最优的数据加载配置。

## 功能特性

- 🔍 **系统监控**: 实时监控 CPU、GPU、内存、磁盘 I/O 和网络 I/O
- 📊 **性能分析**: 测量训练吞吐量和延迟
- 🎯 **智能优化**: 基于多维度指标自动选择最优 `num_workers` 设置
- 📈 **可视化报告**: 生成详细的性能图表和 CSV 报告
- ⚡ **自动化测试**: 自动测试多个 `num_workers` 值并比较结果

## 安装依赖

```bash
pip install -r requirements_optimize.txt
```

## 使用方法

### 基本用法

```bash
python optimize_workers.py --cfg ../configs/butterflyfishes.yaml
```

### 高级选项

```bash
python optimize_workers.py \
    --cfg ../configs/butterflyfishes.yaml \
    --max-workers 16 \
    --epochs-per-test 2
```

### 参数说明

- `--cfg`: 训练配置文件路径（必需）
- `--max-workers`: 测试的最大 worker 数量（默认: 16）
- `--epochs-per-test`: 每个测试运行的 epoch 数（默认: 2）

## 测试流程

脚本会自动测试以下 `num_workers` 值：
- 0, 1, 2, 4, 8, 12, 16（不超过 `--max-workers`）

对于每个值，脚本会：
1. 修改配置文件中的 `num_workers` 参数
2. 启动系统资源监控
3. 运行训练过程（禁用验证以加快速度）
4. 收集性能指标
5. 计算综合评分

## 输出结果

### 控制台输出
```
============================================================
Testing with num_workers = 4
============================================================
Results for num_workers=4:
  Total time: 45.23s
  Throughput: 44.22 samples/sec
  Max CPU: 85.2%
  Max Memory: 67.8%
  Max GPU: 92.1%

============================================================
OPTIMAL SETTING: num_workers = 8
Score: 0.847
Throughput: 52.34 samples/sec
Avg CPU: 78.5%
Avg Memory: 65.2%
============================================================
```

### 生成文件

1. **CSV 报告**: `worker_optimization_results.csv`
   - 包含所有测试结果的详细数据
   - 可用于进一步分析

2. **性能图表**: `worker_optimization_plots.png`
   - 吞吐量 vs num_workers
   - CPU 使用率 vs num_workers
   - 内存使用率 vs num_workers
   - GPU 利用率 vs num_workers

## 评分算法

最优设置基于以下加权评分：

```
Score = 0.6 × 吞吐量得分 + 0.2 × CPU效率 + 0.2 × 内存效率
```

其中：
- **吞吐量得分**: 相对于最高吞吐量的归一化值
- **CPU效率**: 1 - (平均CPU使用率 / 100)
- **内存效率**: 1 - (平均内存使用率 / 100)

## 注意事项

1. **测试时间**: 每个 `num_workers` 值需要运行 `--epochs-per-test` 个 epoch，总时间取决于数据集大小和硬件配置

2. **资源占用**: 测试过程中会占用大量系统资源，建议在空闲时运行

3. **配置文件**: 脚本会临时修改配置文件，测试完成后会恢复原设置

4. **GPU 要求**: 需要 NVIDIA GPU 和 CUDA 环境来监控 GPU 使用率

5. **权限要求**: 需要系统监控权限来收集资源使用数据

## 故障排除

### 常见问题

1. **导入错误**: 确保已安装所有依赖包
   ```bash
   pip install -r requirements_optimize.txt
   ```

2. **权限错误**: 在 Linux 上可能需要 sudo 权限
   ```bash
   sudo python optimize_workers.py --cfg config.yaml
   ```

3. **GPU 监控失败**: 检查 NVIDIA 驱动和 GPUtil 安装
   ```bash
   pip install --upgrade GPUtil
   ```

4. **配置文件错误**: 确保配置文件路径正确且格式有效

### 调试模式

如需查看详细日志，可以修改脚本中的 stdout 重定向部分：

```python
# 注释掉这行来显示训练输出
# sys.stdout = open(os.devnull, 'w')
```

## 示例结果

典型的优化结果可能如下：

| num_workers | 吞吐量 (samples/sec) | 平均 CPU (%) | 平均内存 (%) | 评分 |
|-------------|---------------------|-------------|-------------|------|
| 0           | 25.6                | 45.2        | 35.8        | 0.623 |
| 1           | 32.1                | 52.4        | 38.9        | 0.701 |
| 2           | 38.7                | 61.3        | 42.1        | 0.784 |
| 4           | 44.2                | 78.5        | 65.2        | 0.847 |
| 8           | 52.3                | 85.7        | 72.8        | 0.892 |
| 12          | 51.8                | 89.2        | 78.4        | 0.856 |
| 16          | 50.1                | 92.6        | 82.1        | 0.823 |

在这个例子中，`num_workers = 8` 是最优设置。 