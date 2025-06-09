#!/usr/bin/env python3
"""
yolov5_replace_act_inspector.py

1) Load a YOLOv5 FP32 model (either from torch.hub or a local .pt).
2) Recursively replace every nn.SiLU (Swish) with nn.ReLU.
3) Run NNDCT Inspector on the modified model to see which ops now map to the DPU.

Usage:
    pip install torch torchvision pytorch_nndct
    # If you also need torch.hub to pull in YOLOv5 code, you need internet once.
    python yolov5_replace_act_inspector.py \
        --model yolov5s.pt \
        --target DPUCZDX8G_ISA0 \
        [--img_size 640]

Arguments:
    --model     Path to your YOLOv5 .pt weights (e.g. yolov5s.pt) or built-in name like "yolov5s".
    --target    NNDCT target string (e.g. "DPUCZDX8G_ISA0").
    --img_size  Input size (YOLOv5 default is 640). Change if your model uses a different resolution.
"""

import argparse
import torch
import torch.nn as nn
from pytorch_nndct.apis import Inspector

def parse_args():
    p = argparse.ArgumentParser(
        description="Replace unsupported SiLU→ReLU in YOLOv5, then run NNDCT Inspector."
    )
    p.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "YOLOv5 weights file (e.g. yolov5s.pt) or a built-in name "
            "like 'yolov5s' to fetch pretrained via torch.hub."
        )
    )
    p.add_argument(
        "--target",
        type=str,
        default="DPUCZDX8G_ISA1_B4096",
        help="NNDCT target_name for your DPU (e.g. DPUCZDX8G_ISA0)."
    )
    p.add_argument(
        "--img_size",
        type=int,
        default=640,
        help="Input size (YOLOv5 default is 640). Change if your model uses another resolution."
    )
    return p.parse_args()


def load_yolov5_model(model_arg, device):
    """
    Load a YOLOv5 FP32 model from torch.hub or a local .pt. Returns the raw nn.Module.
    """
    print(f"[INFO] Loading YOLOv5 model '{model_arg}' …")
    if model_arg.lower().endswith(".pt"):
        # Custom local checkpoint
        model_wrapper = torch.hub.load(
            'ultralytics/yolov5',    # Repo to clone
            'custom',                # Load custom weights
            path=model_arg,
            force_reload=True        # Re-clone if needed
        )
    else:
        # Built-in name: "yolov5s", "yolov5m", etc.
        model_wrapper = torch.hub.load(
            'ultralytics/yolov5',
            model_arg,
            pretrained=True
        )
    pt_model = model_wrapper.model  # The raw nn.Module inside the wrapper
    pt_model.to(device).eval()
    return pt_model


def replace_silu_with_relu(module: nn.Module) -> None:
    """
    Recursively walk `module` and swap every nn.SiLU (Swish) with nn.ReLU(inplace=True).
    This edits `module` in place (no return).  
    """
    for name, child in module.named_children():
        # If the child *is* a SiLU, replace it with ReLU
        if isinstance(child, nn.SiLU):
            setattr(module, name, nn.ReLU(inplace=True))
        else:
            # Otherwise recurse into the child's submodules
            replace_silu_with_relu(child)


def main():
    args = parse_args()

    # 1) Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 2) Load YOLOv5 FP32
    pt_model = load_yolov5_model(args.model, device)

    # 3) Before replacement, you can inspect how many SiLU layers exist:
    silu_count = sum(1 for m in pt_model.modules() if isinstance(m, nn.SiLU))
    print(f"[INFO] Found {silu_count} nn.SiLU modules before replacement.")

    # 4) Replace all SiLU → ReLU
    #replace_silu_with_relu(pt_model)

    # 5) Count again to verify
    silu_count_after = sum(1 for m in pt_model.modules() if isinstance(m, nn.SiLU))
    relu_count = sum(1 for m in pt_model.modules() if isinstance(m, nn.ReLU))
    print(f"[INFO] Found {silu_count_after} nn.SiLU modules after replacement (should be 0).")
    print(f"[INFO] Found {relu_count} nn.ReLU modules in total.")

    # 6) Create dummy input
    dummy_input = torch.randn(1, 3, args.img_size, args.img_size).to(device)

    # 7) Run NNDCT Inspector
    print(f"[INFO] Running NNDCT Inspector on modified model (SiLU→ReLU) for target '{args.target}' …")
    inspector = Inspector(args.target)
    inspector.inspect(pt_model, (dummy_input,), device=device)

    print("\n[INFO] Inspector report generated in './__inspect__/'.")
    print("       ├─ inspect.json")
    print("       ├─ inspect.csv")
    print("       └─ layerwise_debug.txt")
    print("[INFO] You can open inspect.json or load inspect.csv into Excel to review DPU vs CPU mapping.")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()