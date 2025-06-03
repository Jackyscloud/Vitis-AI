import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse


def parse_args():
    p = argparse.ArgumentParser(
        description="Modify YOLOv5 .pt: replace unsupported activations + convert weights to NHWC, then save new .pt."
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
        "--img_size",
        type=int,
        default=640,
        help="Input image size (YOLOv5 default is 640). Change if you trained at another resolution."
    )
    p.add_argument(
        "--output",
        type=str,
        default="yolov5_nhwc_relu_hsigmoid.pt",
        help="Path to save the modified .pt file."
    )
    return p.parse_args()


# a) 將所有 nn.SiLU 換成 nn.ReLU
def replace_silu_with_relu(m: nn.Module):
    for name, child in m.named_children():
        if isinstance(child, nn.SiLU):
            setattr(m, name, nn.ReLU(inplace=True))
        else:
            replace_silu_with_relu(child)


# b) 將所有 nn.Sigmoid 換成 HardSigmoid
class HardSigmoid(nn.Module):
    def forward(self, x):
        # 用 ReLU6(x+3)/6 作為近似
        return F.relu6(x + 3, inplace=True) / 6.0


def replace_sigmoid_with_hardsigmoid(m: nn.Module):
    for name, child in m.named_children():
        if isinstance(child, nn.Sigmoid):
            setattr(m, name, HardSigmoid())
        else:
            replace_sigmoid_with_hardsigmoid(child)


# c) 將所有 nn.Softmax 換成「直接輸出 logits」
class NoOpSoftmax(nn.Module):
    def forward(self, x):
        return x  # 不做任何操作，保留 logits


def replace_softmax_with_noop(m: nn.Module):
    for name, child in m.named_children():
        if isinstance(child, nn.Softmax):
            setattr(m, name, NoOpSoftmax())
        else:
            replace_softmax_with_noop(child)


# 3. 僅把「rank >=4 的參數和 buffer」轉成 channels_last
def convert_model_to_nhwc(model: nn.Module):
    model.eval()
    # 參數 (weights) 轉成 channels_last
    for p in model.parameters():
        if p.ndimension() >= 4:
            p.data = p.data.to(memory_format=torch.channels_last)
    # buffer (如 BatchNorm 的 running_mean/var) 也轉
    for b in model.buffers():
        if b.ndimension() >= 4:
            b.data = b.data.to(memory_format=torch.channels_last)
    # 不直接呼叫 model.to(memory_format=channels_last)，以免觸發 rank<4 的張量出錯
    return model


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 載入 YOLOv5 .pt
    print(f"[INFO] Loading YOLOv5 model '{args.model}' …")
    if args.model.lower().endswith(".pt"):
        # local checkpoint
        yolo5_wrapper = torch.hub.load(
            'ultralytics/yolov5',
            'custom',
            path=args.model,
            force_reload=True
        )
        pt_model = yolo5_wrapper.model
    else:
        # 抓官方 pretrained
        yolo5_wrapper = torch.hub.load(
            'ultralytics/yolov5',
            args.model,  # e.g. 'yolov5s'
            pretrained=True
        )
        pt_model = yolo5_wrapper.model

    pt_model.to(device).eval()

    # 2. 修改 activation：SiLU→ReLU、Sigmoid→HardSigmoid、Softmax→NoOp
    replace_silu_with_relu(pt_model)
    replace_sigmoid_with_hardsigmoid(pt_model)
    replace_softmax_with_noop(pt_model)

    # 3. 將 rank>=4 的 tensor(權重、BN 之類) 轉為 channels_last (NHWC)
    pt_model = convert_model_to_nhwc(pt_model)

    # 4. 用 NHWC 的假輸入測 forward，確認 shape 正確
    dummy = torch.randn(1, 3, args.img_size, args.img_size).to(memory_format=torch.channels_last).to(device)
    with torch.no_grad():
        out = pt_model(dummy)
    print("[INFO] Forward OK，輸出大小：", out.shape)

    # 5. 把整個模型儲存成新的 .pt
    #    如果你想保存帶 wrapper 的整個 model，可以直接 torch.save(yolo5_wrapper, ...)
    #    但為了便於後續用 Vitis AI 量化工具讀入，我們只存 state_dict
    torch.save({'model_state_dict': pt_model.state_dict()}, args.output)
    print(f"[INFO] 已將修改後的模型儲存到：{args.output}")

    # 註：後續如果要在 Vitis AI PyTorch Quantizer/Inspector 階段使用這個 .pt，
    #     可以像以下範例一樣呼叫：
    #
    # vai_q_pytorch --model_pt yolov5_nhwc_relu_hsigmoid.pt \
    #               --input_nodes images --output_nodes output \
    #               --input_shapes 1,3,{img_size},{img_size} \
    #               --dump_inspect quant_inspect_dir
    #
    # 或者在 Python 內用 Xilinx NNDCT 的 API 直接載入 inspect：
    #   from nndct_shared import inspector
    #   inspector.run_inspector('yolov5_nhwc_relu_hsigmoid.pt', target='DPUCZDX8G_ISA1_B4096', img_size={img_size})
    #

if __name__ == "__main__":
    main()