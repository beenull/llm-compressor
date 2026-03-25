"""
Qwen3-VL 2B (内部 Omni 模型): QuaRot (R1+R2+R4) + GPTQ W4 量化脚本

使用内部 OmniQwen3VLMedusaModel 构建流程加载模型 (build_model)，
对其中的 Qwen3VLForConditionalGeneration 骨干做 QuaRot+GPTQ 量化，
量化后运行 ATQTA multi-turn 推理验证正确性。

Usage:
    # 静默模式
    python examples/qwen3vl_quant/quarot_gptq.py --skip-quant

    # 调试模式 (输出全部日志)
    DEBUG=true python examples/qwen3vl_quant/quarot_gptq.py \
        --omni-model-name /workspace/gaoy25@xiaopeng.com/model/qwen3_vl/Qwen3-VL-2B-Instruct \
        --omni-model-tokenizer /workspace/gaoy25@xiaopeng.com/model/group_share/adc-perception-mlinfra/shijh2/qwen3_vl_extend \
        --omni-model-ckpt /workspace/gaoy25@xiaopeng.com/model/group_share/adc-perception-mlinfra/malf/omni/hf2aif_0304_final_resave.pt \
        --device cuda:0
"""

import argparse
import logging
import os
import sys

import torch
from datasets import load_dataset

from llmcompressor import oneshot
from llmcompressor.modifiers.transform.spinquant import mappings, norm_mappings
from llmcompressor.modifiers.autoround import AutoRoundModifier
from llmcompressor.utils import dispatch_for_generation

# DEBUG 模式: DEBUG=true python ... 时才输出调试信息
DEBUG = os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")

if not DEBUG:
    logging.disable(logging.INFO)


def dprint(*args, **kwargs):
    """仅在 DEBUG=true 时打印"""
    if DEBUG:
        print(*args, **kwargs)

# 本地依赖 (从 xllm-evaluation 拷贝)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_model import build_model
from build_prompt import build_chat_prompt


#################### configurations ####################
recipe = "examples/qwen3vl_quant/configs/quarot_gptq.yaml"
model_dtype = torch.bfloat16

NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 2048

SAVE_DIR = "/tmp/Qwen3-VL-2B-quarot-gptq-w4"
#################### configurations ####################


# =====================================================
# 注册 Qwen3-VL SpinQuant 映射 (与 Qwen2.5-VL 路径一致)
# =====================================================
mappings.SPINQUANT_MAPPING_REGISTRY["Qwen3VLForConditionalGeneration"] = (
    mappings.SpinQuantMapping(
        mm_proj=[r"re:.*visual\.merger.*linear_fc2$"],
        embedding="re:.*embed_tokens$",
        attn="re:.*self_attn$",
        attn_q="re:.*language_model.*q_proj$",
        attn_k="re:.*language_model.*k_proj$",
        attn_v="re:.*language_model.*v_proj$",
        attn_o="re:.*language_model.*o_proj$",
        mlp_in=[
            r"re:.*language_model.*mlp\.up_proj$",
            r"re:.*language_model.*mlp\.gate_proj$",
        ],
        mlp_out=[r"re:.*language_model.*mlp\.down_proj$"],
        lm_head="lm_head",
    )
)

norm_mappings.NORM_MAPPING_REGISTRY["Qwen3VLForConditionalGeneration"] = [
    norm_mappings.NormMapping(
        norm="re:.*language_model.*input_layernorm$",
        linears=[
            "re:.*language_model.*q_proj$",
            "re:.*language_model.*k_proj$",
            "re:.*language_model.*v_proj$",
        ],
    ),
    norm_mappings.NormMapping(
        norm="re:.*language_model.*post_attention_layernorm$",
        linears=[
            r"re:.*language_model.*mlp\.up_proj$",
            r"re:.*language_model.*mlp\.gate_proj$",
        ],
    ),
    norm_mappings.NormMapping(
        norm="model.language_model.norm",
        linears=["lm_head"],
    ),
]

# =====================================================
# 主流程
# =====================================================
def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL QuaRot+GPTQ quantization")
    parser.add_argument(
        "--omni-model-name",
        type=str,
        default="/workspace/gaoy25@xiaopeng.com/model/qwen3_vl/Qwen3-VL-2B-Instruct",
    )
    parser.add_argument(
        "--omni-model-tokenizer",
        type=str,
        default="/workspace/gaoy25@xiaopeng.com/model/group_share/adc-perception-mlinfra/shijh2/qwen3_vl_extend",
    )
    parser.add_argument(
        "--omni-model-ckpt",
        type=str,
        default="/workspace/gaoy25@xiaopeng.com/model/group_share/adc-perception-mlinfra/malf/omni/hf2aif_0304_final_resave.pt",
        
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--skip-quant", action="store_true", help="跳过量化，只做推理测试")
    args = parser.parse_args()

    # =================================================
    # Step 1: 用内部 build_model 构建 OmniQwen3VLMedusaModel
    # =================================================
    dprint(f"[Step 1] Building Omni model from {args.omni_model_name}...")
    omni_model, tokenizer = build_model(
        omni_model_name=args.omni_model_name,
        omni_model_tokenizer_path=args.omni_model_tokenizer,
        omni_model_ckpt_path=args.omni_model_ckpt,
        device=args.device,
    )

    # 提取内部的 Qwen3VLForConditionalGeneration 用于量化
    # omni_model.language_model 就是 Qwen3VLForConditionalGeneration
    hf_model = omni_model.language_model

    # =================================================
    # Step 2: 量化前先跑 ATQTA 推理 (baseline 对比)
    # =================================================
    dprint("\n[Step 2] Running ATQTA inference BEFORE quantization...")
    run_atqta_infer(omni_model, tokenizer, args.device)

    if args.skip_quant:
        dprint("[INFO] --skip-quant specified, skipping quantization.")
        return


    # =================================================
    # Step 3: 准备校准数据 (支持 text/vit 多模态)
    # =================================================
    dprint("[Step 3] Preparing calibration data...")

    from .task_utils import get_multimodal_calib_dataset, preprocess_for_chat_template, tokenize_for_calib

    enable_modality = {
        # "vit",
        "text"
    }
    HF_DATASETS_CACHE = "/dataset/workspace/zhangl98/hf_cache/"

    ds = get_multimodal_calib_dataset(NUM_CALIBRATION_SAMPLES, enable_modality, HF_DATASETS_CACHE)
    ds = ds.map(lambda example: preprocess_for_chat_template(example, tokenizer))
    ds = ds.map(lambda sample: tokenize_for_calib(sample, tokenizer, MAX_SEQUENCE_LENGTH), remove_columns=ds.column_names)

    # =================================================
    # Step 4: 执行 QuaRot + GPTQ oneshot 量化
    # =================================================
    dprint("[Step 4] Running QuaRot + GPTQ quantization...")
    oneshot(
        model=hf_model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        shuffle_calibration_samples=False,
    )

    # =================================================
    # Step 5: 量化后跑 ATQTA 推理 (验证量化正确性)
    # =================================================
    dprint("\n[Step 5] Running ATQTA inference AFTER quantization...")
    run_atqta_infer(omni_model, tokenizer, args.device)

    # =================================================
    # Step 6: 保存量化模型
    # =================================================
    dprint(f"[Step 6] Saving quantized model to {SAVE_DIR}...")
    from llmcompressor.transformers.compression.compressed_tensors_utils import (
        modify_save_pretrained,
    )

    modify_save_pretrained(hf_model)
    hf_model.save_pretrained(SAVE_DIR, save_compressed=True)
    tokenizer.save_pretrained(SAVE_DIR)
    dprint(f"Quantized model saved to {SAVE_DIR}")


if __name__ == "__main__":
    main()
