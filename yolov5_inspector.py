#!/usr/bin/env python3
"""
yolov5_inspector.py

A standalone script to run Xilinx’s NNDCT Inspector on a YOLOv5 FP32 model
and report which layers would map to the DPU versus fallback to CPU.

Usage:
    pip install torch torchvision pytorch_nndct
    # The first time you run this, TorchHub will clone the YOLOv5 repo automatically:
    python yolov5_inspector.py \
        --model yolov5s.pt \
        --target DPUCZDX8G_ISA0 \
        [--img_size 640]

Arguments:
    --model     Path to your YOLOv5 .pt weights (e.g. yolov5s.pt). You can also pass
                a built‐in name like "yolov5s" if you want the official pretrained.
    --target    NNDCT target name for your DPU (e.g. DPUCZDX8G_ISA0).
    --img_size  (Optional) Input size for YOLOv5 (default: 640). If your network
                was trained at a different resolution (e.g. 512), set this accordingly.
"""

import argparse
import torch
from pytorch_nndct.apis import Inspector

def parse_args():
    p = argparse.ArgumentParser(
        description="Run NNDCT Inspector on a YOLOv5 FP32 model (no calibration/quant loops)."
    )
    p.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "Path to YOLOv5 .pt weights (e.g. yolov5s.pt), "
            "or a built‐in repo name like 'yolov5s' to fetch the official pretrained."
        )
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
        help="Input image size (YOLOv5 default is 640). Change if you trained at another resolution."
    )
    return p.parse_args()


def main():
    args = parse_args()

    # 1) Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 2) Load the YOLOv5 FP32 model via TorchHub (ultralytics/yolov5).
    #    If args.model == "yolov5s" (or "yolov5m"/"yolov5l"/"yolov5x"), TorchHub will fetch the official pretrained.
    #    Otherwise, if args.model is a path ending with ".pt", we treat it as a custom weight file.
    print(f"[INFO] Loading YOLOv5 model '{args.model}' …")
    if args.model.lower().endswith(".pt"):
        # Load a custom local checkpoint
        #   torch.hub.load(..., "custom", path=PATH_TO_PT)
        yolo5_wrapper = torch.hub.load(
            'ultralytics/yolov5',    # repo name
            'custom',                # tells hub to load custom weights
            path=args.model,         # local path to your .pt
            force_reload=True        # re‐clone repo if needed
        )
    else:
        # Load official pretrained by name: "yolov5s", "yolov5m", etc.
        yolo5_wrapper = torch.hub.load(
            'ultralytics/yolov5',
            args.model,              # e.g. "yolov5s", "yolov5m", ...
            pretrained=True
        )

    # The hub returns a wrapper whose `.model` attribute is the raw nn.Module (the Detect class).
    pt_model = yolo5_wrapper.model
    pt_model.to(device).eval()

    # 3) Create a dummy input matching YOLOv5’s expected shape
    #    By default, YOLOv5 “s” uses 640×640; change if you trained at a different resolution.
    dummy_input = torch.randn(1, 3, args.img_size, args.img_size).to(device)

    # 4) Instantiate and run the Inspector
    print(f"[INFO] Running NNDCT Inspector for target '{args.target}' …")
    inspector = Inspector(args.target)

    # The call below will:
    #   - Trace the FP32 graph (including backbone, neck, head, and NMS if present).
    #   - Perform shape inference on each node.
    #   - Compare each op in the graph against the DPU’s supported op set.
    #   - Dump a per-layer report into ./__inspect__/
    inspector.inspect(pt_model, (dummy_input,), device=device)

    # 5) Final status
    print("\n[INFO] Inspector report generated in directory './__inspect__/'.")
    print("       ├─ inspect.json       (per-layer mapping to DPU or CPU fallback)")
    print("       ├─ inspect.csv        (same info in CSV form)")
    print("       └─ layerwise_debug.txt  (detailed debug log)")
    print("[INFO] You can open inspect.json or load inspect.csv into Excel to review which layers map to DPU vs. CPU.")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()