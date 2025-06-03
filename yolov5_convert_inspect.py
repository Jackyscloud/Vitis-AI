import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from pytorch_nndct.apis import Inspector

# -------------------------
# 1. Parser
# -------------------------
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


# -------------------------
# 2. Activation 替换
# -------------------------
def replace_silu_with_relu(m: nn.Module):
    for name, child in m.named_children():
        if isinstance(child, nn.SiLU):
            setattr(m, name, nn.ReLU(inplace=True))
        else:
            replace_silu_with_relu(child)


class HardSigmoid(nn.Module):
    def forward(self, x):
        return F.relu6(x + 3, inplace=True) / 6.0


def replace_sigmoid_with_hardsigmoid(m: nn.Module):
    for name, child in m.named_children():
        if isinstance(child, nn.Sigmoid):
            setattr(m, name, HardSigmoid())
        else:
            replace_sigmoid_with_hardsigmoid(child)


class NoOpSoftmax(nn.Module):
    def forward(self, x):
        return x


def replace_softmax_with_noop(m: nn.Module):
    for name, child in m.named_children():
        if isinstance(child, nn.Softmax):
            setattr(m, name, NoOpSoftmax())
        else:
            replace_softmax_with_noop(child)


# -------------------------
# 3. NHWC 转换（只处理 rank>=4 的 tensor）
# -------------------------
def convert_model_to_nhwc(model: nn.Module):
    model.eval()
    for p in model.parameters():
        if p.ndimension() >= 4:
            p.data = p.data.to(memory_format=torch.channels_last)
    for b in model.buffers():
        if b.ndimension() >= 4:
            b.data = b.data.to(memory_format=torch.channels_last)
    return model


# -------------------------
# 4. 主流程
# -------------------------
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4.1 加载模型
    print(f"[INFO] Loading YOLOv5 model '{args.model}' ...")
    if args.model.lower().endswith(".pt"):
        # 如果本地有 yolov5s.pt，直接加载，force_reload 确保仓库是最新的
        yolo5_wrapper = torch.hub.load(
            'ultralytics/yolov5',
            'custom',
            path=args.model,
            force_reload=True,
            trust_repo=True
        )
        pt_model = yolo5_wrapper.model
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

    # 4.2 替换不支持的 Activation: SiLU->ReLU, Sigmoid->HardSigmoid, Softmax->NoOp
    replace_silu_with_relu(pt_model)
    replace_sigmoid_with_hardsigmoid(pt_model)
    replace_softmax_with_noop(pt_model)

    # 4.3 转 NHWC：只对 rank>=4 的 tensor（conv weight、BN stats）做 memory_format=torch.channels_last
    model = convert_model_to_nhwc(pt_model)

    # 4.4 用 NHWC 假输入做一次 forward 验证
    dummy = torch.randn(1, 3, args.img_size, args.img_size) \
                .to(memory_format=torch.channels_last).to(device)
    with torch.no_grad():
        out = model(dummy)
    print("[INFO] Forward OK, output shape:", out.shape)

    # 4.5 调用 NNDCT Inspector 检查 DPU unsupported nodes
    #     假设你已安装好 Vitis AI PyTorch Quantizer (nndct)
    inspector = Inspector(args.target)

    inspector.inspect(model, (dummy,), device=device)

    # 5) Final status
    print("\n[INFO] Inspector report generated in directory './__inspect__/'.")
    print("       ├─ inspect.json       (per-layer mapping to DPU or CPU fallback)")
    print("       ├─ inspect.csv        (same info in CSV form)")
    print("       └─ layerwise_debug.txt  (detailed debug log)")
    print("[INFO] You can open inspect.json or load inspect.csv into Excel to review which layers map to DPU vs. CPU.")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()