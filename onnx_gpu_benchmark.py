# Basic lib
import json
import os
import time
import yaml
from pathlib import Path

# DL lib
import onnx
import onnxruntime
import numpy as np
from PIL import Image
import torch
import torch.onnx

# Local lib
from models.utils import Model
from models.utils import set_deterministic
from fam_cls_train import load_pretrained_ckpt
from backbone_info_benchmark import image_preprocess


def onnx_inference_time_test(ort_session, ort_inputs, num_warmup_iters=10, num_iters=100):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Warmup
    for _ in range(num_warmup_iters):
        _ = ort_session.run(['output'], ort_inputs)[0]

    # Sync the asynchronous execution of cpu and gpu, cpu wait for the gpu to complete all tasks
    torch.cuda.synchronize() if device.type == 'cuda' else None

    # ORT GPU inference
    inference_times = []
    for i in range(num_iters):
        time_s = time.time()
        _ = ort_session.run(['output'], ort_inputs)[0]
        torch.cuda.synchronize() if device.type == 'cuda' else None
        time_e = time.time()
        inference_times.append((time_e - time_s) * 1000)  # ms

    # Inference time statistics
    avg_time = np.mean(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    std_time = np.std(inference_times)

    print(f"\n=== GPU ONNX Runtime Inference Time ===")
    print(f"Avg: {avg_time:.2f} ms")
    print(f"Min: {min_time:.2f} ms")
    print(f"Max: {max_time:.2f} ms")
    print(f"Std: {std_time:.2f} ms")
    print(f"FPS: {1000 / avg_time:.2f}")

    return {
        'avg_time_ms': avg_time,
        'min_time_ms': min_time,
        'max_time_ms': max_time,
        'std_time_ms': std_time,
        'fps': 1000 / avg_time
    }


def onnx_classify_test(ort_session, ort_inputs, imagenet_classes_name):
    """
    Test image classification using ImageNet pretrained weights.
    Tests all four models: resnet50, resnet101, vgg16, vgg19.
    """
    # Perform classification
    with torch.no_grad():
        ort_outputs = ort_session.run(['output'], ort_inputs)[0]
        probabilities = torch.softmax(torch.from_numpy(ort_outputs), dim=1)

        # Get top-5 predictions
        top5_prob, top5_indices = torch.topk(probabilities, 5, dim=1)
        print(f"Top-5 Predictions:")

        for i in range(5):
            class_idx = top5_indices[0, i].item()
            prob = top5_prob[0, i].item()
            class_name = imagenet_classes_name[class_idx]
            print(f"{i:2d}. {class_name:<30} {prob:.4f}")

        # Show top prediction
        top1_idx = top5_indices[0, 0].item()
        top1_prob = top5_prob[0, 0].item()
        top1_class = imagenet_classes_name[top1_idx]

        print(f"\n=== ImageNet Classification ===")
        print(f"\n🎯 Predicted Class: {top1_class}")
        print(f"   Confidence: {top1_prob:.4f}")

        return {
            "pred_class": top1_class,
            "confidence": top1_prob,
        }


def main():
    print("=== GPU ONNX Inference Benchmark Test ===")

    # Test data
    image_path = "Tocal.jpg"
    image_tensor = image_preprocess(image_path)

    # Load ImageNet class names
    with open("configs/imagenet_class_index.json", "r") as f:
        imagenet_classes = json.load(f)
    imagenet_classes_name = [imagenet_classes[f"{i}"][1] for i in range(len(imagenet_classes))]

    # Export ONNX models
    model_names_list = ["resnet50", "resnet101", "vgg16", "vgg19"]
    for model_name in model_names_list:
        # Construct model by the config file
        model_cfg_path = os.path.join("configs", f"{model_name}.yaml")
        with open(model_cfg_path, 'r', encoding='utf-8') as f:
            model_cfg = yaml.safe_load(f)
        model = Model(model_cfg)

        # Load pretrained weights
        pretrained_weight_path = Path("pretrained", f"{model_name}_pretrained.pth")
        model = load_pretrained_ckpt(model, pretrained_weight_path)

        # Export ONNX model
        with torch.no_grad():
            torch.onnx.export(
                model,
                image_tensor,
                f"onnx/{model_name}.onnx",
                opset_version=11,
                input_names=['input'],
                output_names=['output'])


    # ONNX Runtime Inference Test
    for model_name in model_names_list:
        print(f"\n=== Testing {model_name} ===")
        # Check ONNX Model
        onnx_model = onnx.load("onnx/%s.onnx" % model_name)
        onnx.checker.check_model(onnx_model)

        # ONNX Runtime session
        ort_session = onnxruntime.InferenceSession("onnx/%s.onnx" % model_name, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        ort_inputs = {'input': image_tensor.detach().numpy()}

        # ONNX Runtime GPU inference time server
        time_test_results = onnx_inference_time_test(ort_session, ort_inputs)

        # ONNX GPU classification server
        classify_results = onnx_classify_test(ort_session, ort_inputs, imagenet_classes_name)

        # Save report
        with open(f"results/{model_name}_report.txt", "a") as f:
            f.write("\n")
            f.write("=" * 100)
            f.write("\nDevice: GPU (RTX-4090)\n")
            f.write("Framework: ONNX Runtime\n")
            f.write("=" * 100)
            f.write("\nClassification Test\n")
            for key, value in classify_results.items():
                f.write(f"{key}: {value}\n")
            f.write("=" * 100)
            f.write("\nInference Time Test\n")
            for key, value in time_test_results.items():
                f.write(f"{key}: {value:.4f}\n")
            f.write("=" * 100)


if __name__ == "__main__":
    set_deterministic(random_seed=666)
    main()