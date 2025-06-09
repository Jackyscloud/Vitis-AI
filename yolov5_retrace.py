import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

def parse_args():
    p = argparse.ArgumentParser(
        description="Run NNDCT Inspector on a YOLOv5 .pt model (NHWC + replace unsupported activations)."
    )
    p.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "Path to YOLOv5 .pt (e.g. yolov5s.pt), "
            "or a built-in repo name like 'yolov5s' to fetch official pretrained."
        )
    )
    p.add_argument(
        "--img_size",
        type=int,
        default=640,
        help="Input image size (YOLOv5 default 640)."
    )
    p.add_argument(
        "--target",
        type=str,
        default="DPUCZDX8G_ISA1_B4096",
        help="NNDCT target name (e.g. DPUCZDX8G_ISA0)."
    )
    return p.parse_args()


def main():

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4.1 加载模型
    print(f"[INFO] Loading YOLOv5 model '{args.model}' ...")
    if args.model.lower().endswith(".pt"):
        # 如果本地有 yolov5s.pt，直接加载，force_reload 确保仓库是最新的
        yolo5_wrapper = torch.hub.load(
            'yolov5-master',
            'custom',
            path=args.model,
            source ='local',
            force_reload=True,
            trust_repo=True
        )
        pt_model = yolo5_wrapper.model

        print(f"[INFO] Model loaded from {args.model} with {len(pt_model)} layers.")
    else:
        # 直接在线加载官方 pretrained，比如 'yolov5s','yolov5m' 等
        yolo5_wrapper = torch.hub.load(
            'ultralytics/yolov5',
            args.model,
            pretrained=True,
            force_reload=True,
            trust_repo=True
        )
        pt_model = yolo5_wrapper.model

    pt_model.to(device).eval()

    scripted = torch.jit.script(pt_model)         # or model.export() if using ultralytics export
    scripted.save("yolov5_hardware.pt")


if __name__ == "__main__":
    main()
