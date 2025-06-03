#!/usr/bin/env python3
"""
yolov8_inspector.py

A standalone script to run Xilinx’s NNDCT Inspector on a YOLOv8 FP32 model
and report which layers would map to the DPU versus fallback to CPU.

Usage:
    pip install torch torchvision ultralytics pytorch_nndct
    python yolov8_inspector.py \
        --model yolov8s.pt \
        --target DPUCZDX8G_ISA0 \
        [--img_size 640]

Arguments:
    --model     Path to your YOLOv8 .pt weights (e.g. yolov8s.pt).
    --target    NNDCT target name for your DPU (e.g. DPUCZDX8G_ISA0).
    --img_size  (Optional) Input size for YOLOv8 (default: 640). If your network
                was trained at a different resolution (e.g. 512), set this accordingly.
"""

import argparse
import torch
from ultralytics import YOLO
from pytorch_nndct.apis import Inspector

def parse_args():
    p = argparse.ArgumentParser(
        description="Run NNDCT Inspector on a YOLOv8 FP32 model (no calibration/quant loops)."
    )
    p.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to YOLOv8 .pt weights (e.g. yolov8s.pt)."
    )
    p.add_argument(
        "--target",
        type=str,
        required=True,
        help="NNDCT target name for your DPU (e.g. DPUCZDX8G_ISA0)."
    )
    p.add_argument(
        "--img_size",
        type=int,
        default=640,
        help="Input image size (YOLOv8 default is 640). Change if you trained at another resolution."
    )
    return p.parse_args()

def main():
    args = parse_args()

    # 1) Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 2) Load the YOLOv8 FP32 model via Ultralytics
    print(f"[INFO] Loading YOLOv8 model from '{args.model}' …")
    yolo_wrapper = YOLO(args.model)    # Instantiates Ultralytics YOLO wrapper
    pt_model = yolo_wrapper.model      # Extract the raw torch.nn.Module
    pt_model.to(device).eval()

    # 3) Create a dummy input matching YOLOv8’s expected shape
    #    Default for "s" variant is 640×640; modify if needed.
    dummy_input = torch.randn(1, 3, args.img_size, args.img_size).to(device)

    # 4) Instantiate and run the Inspector
    print(f"[INFO] Running NNDCT Inspector for target '{args.target}' …")
    inspector = Inspector(args.target)

    # `inspect` will:
    #   - Trace the FP32 graph
    #   - Perform shape inference on each node
    #   - Compare each op against the DPU’s supported operation set
    #   - Dump a per-layer report into ./__inspect__/
    inspector.inspect(pt_model, (dummy_input,), device=device)

    print("\n[INFO] Inspector report generated in directory './__inspect__/'.")
    print("       ├─ inspect.json")
    print("       ├─ inspect.csv")
    print("       └─ layerwise_debug.txt")
    print("[INFO] You can open inspect.json or load inspect.csv into Excel to review which layers map to DPU vs. CPU.")
    print("[INFO] Done.")

if __name__ == "__main__":
    main()