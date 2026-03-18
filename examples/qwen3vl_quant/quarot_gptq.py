"""
Qwen3-VL 2B: QuaRot (R1+R2+R4) + GPTQ W4 量化脚本

QuaRot 使用固定 Hadamard 旋转矩阵，无需训练。GPTQ 使用校准数据做权重量化。
单卡即可运行。

Usage:
    python examples/qwen3vl_quant/quarot_gptq.py
"""

import torch
from compressed_tensors.offload import dispatch_model
from datasets import load_dataset
from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.transform.spinquant import mappings, norm_mappings

#################### configurations ####################
MODEL_ID = "/dataset/workspace/models/Qwen3-VL-2B-Instruct"  # 替换为实际路径

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
# 加载模型
# =====================================================
print(f"Loading model from {MODEL_ID}...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=model_dtype,
)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)


# =====================================================
# 准备校准数据 (纯文本)
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
# 执行 QuaRot + GPTQ oneshot 量化
# =====================================================
print("Running QuaRot + GPTQ quantization...")
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    shuffle_calibration_samples=False,
)


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
# 保存量化模型
# =====================================================
from llmcompressor.transformers.compression.compressed_tensors_utils import (
    modify_save_pretrained,
)

modify_save_pretrained(model)
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)
print(f"Quantized model saved to {SAVE_DIR}")
