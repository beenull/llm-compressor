"""
Qwen3-VL 2B: OSTQuant PostQuant (Phase 2)

在 OSTQuant 训练后的 transformed model 上应用 R4 旋转 + GPTQ W4 量化。
单卡即可运行。

Usage:
    python examples/qwen3vl_quant/ostquant_postquant.py
"""

import torch
from compressed_tensors.offload import dispatch_model
from datasets import load_dataset
from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.transform.spinquant import mappings, norm_mappings
from llmcompressor.transformers.compression.compressed_tensors_utils import (
    modify_save_pretrained,
)

#################### configurations ####################
# Phase 1 训练输出的 transformed model 路径
MODEL_ID = "/tmp/Qwen3-VL-2B-Instruct-origin-ostquant(text|)-trans"  # 替换为实际路径

recipe = "examples/qwen3vl_quant/configs/r4_gptq.yaml"
model_dtype = torch.bfloat16

NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 2048

SAVE_DIR = "/tmp/Qwen3-VL-2B-ostquant-gptq-w4"
#################### configurations ####################


# =====================================================
# 注册 Qwen3-VL SpinQuant + Norm 映射
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
# 加载 transformed model
# =====================================================
print(f"Loading transformed model from {MODEL_ID}...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=model_dtype,
)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)


# =====================================================
# 准备校准数据
# =====================================================
ds = load_dataset("hkust-nlp/deita-6k-v0", split=f"train[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)


def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }


ds = ds.map(preprocess)


def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)


# =====================================================
# 执行 R4 + GPTQ oneshot 量化
# =====================================================
print("Running R4 + GPTQ quantization on transformed model...")
ori_save_pretrained = model.save_pretrained

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    shuffle_calibration_samples=False,
)

model.save_pretrained = ori_save_pretrained


# =====================================================
# 验证生成
# =====================================================
print("\n========== SAMPLE GENERATION ==============")
dispatch_model(model)
input_ids = tokenizer("Hello, tell me about yourself.", return_tensors="pt").input_ids.to(
    model.device
)
output = model.generate(input_ids, max_new_tokens=50)
print(tokenizer.decode(output[0]))
print("==========================================\n")


# =====================================================
# 保存最终量化模型
# =====================================================
modify_save_pretrained(model)
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)
print(f"Quantized model saved to {SAVE_DIR}")
