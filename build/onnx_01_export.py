#!/usr/bin/env python3
"""Export the trained 2-camera QVGA ACT policy to a single self-contained fp32 ONNX.

This is the one model-specific step of the pipeline. A lerobot policy does not expose a plain
tensor-in / tensor-out function: select_action() hides the action-chunk queue, normalization
lives in the pre/post-processors, and the VAE encoder only exists for training. The wrapper
below selects the deployable subgraph (policy.model on two frames + the state, returning the
action chunk) and hands it to the standard torch.onnx.export. Swapping in a different policy
means rewriting this wrapper; everything downstream (cleanup, bf16, import, torq-compile) is
unchanged.

Input : pretrained_dir with model.safetensors + config.json
Output: image_wrist[1,3,240,320] f32, image_top[1,3,240,320] f32, state[1,6] f32 -> action[1,100,6]
        (normalization is NOT in the graph; the board applies it from norm_params.npz)

Usage:  python onnx_01_export.py /path/to/ckpt -o act_fp32.onnx
"""
import argparse
import os

import onnx
import torch
from lerobot.policies.act.modeling_act import ACTPolicy

H, W = 240, 320


class TwoCamWrapper(torch.nn.Module):
    """policy.model on (wrist, top, state) -> action chunk. Eval mode: no VAE encoder, no action input."""

    def __init__(self, policy):
        super().__init__()
        self.model = policy.model

    def forward(self, image_wrist, image_top, state):
        out = self.model({"observation.images": [image_wrist, image_top], "observation.state": state})
        return out[0] if isinstance(out, tuple) else out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pretrained_dir")
    ap.add_argument("-o", "--out", default="act_fp32.onnx")
    args = ap.parse_args()

    policy = ACTPolicy.from_pretrained(args.pretrained_dir).float().eval()
    D = policy.config.action_feature.shape[0]
    N = policy.config.chunk_size
    wrapped = TwoCamWrapper(policy).eval()

    example = (torch.randn(1, 3, H, W), torch.randn(1, 3, H, W), torch.randn(1, D))
    with torch.no_grad():
        a = wrapped(*example)
    assert tuple(a.shape) == (1, N, D), f"expected action (1,{N},{D}), got {tuple(a.shape)}"
    print(f"forward OK: action {tuple(a.shape)}")

    # dynamo=True emits the token-assembly stack as high-arity Concats that torq-tools cleanup can
    # collapse (the legacy exporter emits a form it cannot). opset 18 keeps LayerNorm as one op;
    # request it explicitly, or the exporter tries (and loudly fails) an opset downgrade first.
    torch.onnx.export(
        wrapped, example, args.out,
        input_names=["image_wrist", "image_top", "state"], output_names=["action"],
        opset_version=18, do_constant_folding=True, dynamo=True,
    )
    # re-save self-contained: downstream tools expect one file, not ONNX external data
    onnx.save_model(onnx.load(args.out), args.out, save_as_external_data=False)
    if os.path.exists(args.out + ".data"):
        os.remove(args.out + ".data")
    print(f"wrote {args.out} (self-contained)")


if __name__ == "__main__":
    main()
