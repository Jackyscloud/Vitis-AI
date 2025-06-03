import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse


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
            "or a built-in repo name like 'yolov5s' to fetch the official pretrained."
        )
    )
    p.add_argument(
        "--target",
        type=str,
        default="DPUCZDX8G_ISA1_B4096",
        help="NNDCT target name for your DPU (e.g. DPUCZDX8G_ISA0)."
    )
    p.add_argument(
        "--img_size",
        type=int,
        default=640,
        help="Input image size (YOLOv5 default is 640). Change if you trained at another resolution."
    )
    return p.parse_args()


# a) 把 SiLU 全部換成 ReLU
def replace_silu_with_relu(m: nn.Module):
    for name, child in m.named_children():
        if isinstance(child, nn.SiLU):
            setattr(m, name, nn.ReLU(inplace=True))
        else:
            replace_silu_with_relu(child)


# b) 把 Sigmoid 全部換成 HardSigmoid
class HardSigmoid(nn.Module):
    def forward(self, x):
        return F.relu6(x + 3, inplace=True) / 6.0


def replace_sigmoid_with_hardsigmoid(m: nn.Module):
    for name, child in m.named_children():
        if isinstance(child, nn.Sigmoid):
            setattr(m, name, HardSigmoid())
        else:
            replace_sigmoid_with_hardsigmoid(child)


# c) 把 Softmax 移除（或改成直接輸出 logits）
class NoOpSoftmax(nn.Module):
    def forward(self, x):
        return x


def replace_softmax_with_noop(m: nn.Module):
    for name, child in m.named_children():
        if isinstance(child, nn.Softmax):
            setattr(m, name, NoOpSoftmax())
        else:
            replace_softmax_with_noop(child)


# 3. 把整個模型轉為 NHWC 佈局
def convert_model_to_nhwc(model: nn.Module):
    model.eval()
    for p in model.parameters():
        if p.ndimension() >= 4:
            p.data = p.data.to(memory_format=torch.channels_last)
    for b in model.buffers():
        if b.ndimension() >= 4:
            b.data = b.data.to(memory_format=torch.channels_last)
    model.to(memory_format=torch.channels_last)
    return model


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 載入你的模型
    print(f"[INFO] Loading YOLOv5 model '{args.model}' …")
    if args.model.lower().endswith(".pt"):
        # Load a custom local checkpoint
        yolo5_wrapper = torch.hub.load(
            'ultralytics/yolov5',    # repo name
            'custom',                # tells hub to load custom weights
            path=args.model,         # local path to your .pt
            force_reload=True        # re-clone repo if needed
        )
        pt_model = yolo5_wrapper.model
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

    # 2. 依序做以下三件事：
    replace_silu_with_relu(pt_model)
    replace_sigmoid_with_hardsigmoid(pt_model)
    replace_softmax_with_noop(pt_model)

    # 3. 把整個模型轉為 NHWC 佈局
    model = convert_model_to_nhwc(pt_model)

    # 4. 用 NHWC 隨機輸入測試 forward
    dummy = torch.randn(1, 3, args.img_size, args.img_size).to(memory_format=torch.channels_last).to(device)
    with torch.no_grad():
        out = model(dummy)
    print("Forward OK，輸出大小：", out.shape)

    # 5. 匯出 ONNX，確保量化器看到的就是 NHWC、沒有 SiLU/Sigmoid/Softmax
    onnx_path = "model_nhwc_hardware_friendly.onnx"
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        do_constant_folding=True,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        example_outputs=out,
    )
    print(f"完成匯出：{onnx_path}")


if __name__ == "__main__":
    main()