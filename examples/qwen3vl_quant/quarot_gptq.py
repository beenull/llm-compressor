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
# ATQTA 推理测试 (验证量化后模型正确性)
# =====================================================
ATQTA_TEST_PROMPT = "<audio_667><audio_993><audio_4217><audio_1133><audio_2141><audio_73><audio_933><audio_2141><audio_2079><audio_1797><audio_265><audio_997><audio_3113><audio_2288><audio_4106><audio_331><audio_776><audio_2639><audio_2639><audio_4911><audio_109><audio_673><audio_1127><audio_4436><audio_1782><audio_799><audio_12><audio_1367><audio_1879><audio_1089><audio_60><audio_262><audio_2160><audio_3839><audio_4371><audio_262><audio_936><audio_3453><audio_1816><audio_2759><audio_262><audio_384><audio_2759><audio_3645><audio_1824><audio_631><audio_497><audio_2779><audio_1055><audio_2470><audio_631><audio_226><audio_2141><audio_2470><audio_2470><audio_226><audio_497><audio_2470><audio_2470><audio_2470><audio_169><audio_148><audio_2141><audio_2758><audio_2493><audio_756><audio_214><audio_2639><audio_3423><audio_1604><audio_273><audio_273><audio_4938><audio_4203><audio_2200><audio_908><audio_9><audio_1295><audio_2019><audio_4575><audio_26><audio_283><audio_2722><audio_1327><audio_1172><audio_18><audio_355><audio_4073><audio_4513><audio_4641><audio_36><audio_36><audio_1904><audio_1085><audio_2785><audio_53><audio_53><audio_2624><audio_4177><audio_3690><audio_53><audio_53><audio_4274><audio_3277><audio_2494><audio_866><audio_818><audio_1049><audio_3833><audio_2342><audio_803><audio_803><audio_4624><audio_3670><audio_4569><audio_384><audio_936><audio_4734><audio_4734><audio_3708><audio_431><audio_631><audio_4922><audio_2141><audio_2470><audio_226><audio_476><audio_2470><audio_2470><audio_2470><audio_935><audio_327><audio_1029><audio_3654><audio_1050><audio_327><audio_214><audio_1050><audio_3959><audio_1097><audio_898><audio_8><audio_1048><audio_1426><audio_1518><audio_8><audio_361><audio_1044><audio_4842><audio_1075><audio_660><audio_660><audio_2639><audio_2051><audio_3956><audio_866><audio_851><audio_3956><audio_1028><audio_4289><audio_905><audio_799><audio_4086><audio_1660><audio_2179><audio_36><audio_950><audio_1236><audio_4217><audio_1535><audio_965><audio_965><audio_2316><audio_1068><audio_2061><audio_866><audio_896><audio_1432><audio_3759><audio_1049><audio_607><audio_866><audio_2264><audio_2537><audio_1027><audio_423><audio_512><audio_3788><audio_1066><audio_1278><audio_423><audio_423><audio_4073><audio_2313><audio_1639><audio_384><audio_936><audio_5080><audio_3670><audio_4673><audio_631><audio_631><audio_1050><audio_2141><audio_1027><audio_118><audio_550><audio_4922><audio_4922><audio_2430><audio_169><audio_191><audio_3708><audio_3190><audio_2440><audio_191><audio_219><audio_3738><audio_1375><audio_2973><audio_231><audio_231><audio_1599><audio_1986><audio_1052><audio_231><audio_441><audio_2124><audio_1367><audio_1367><audio_85><audio_76><audio_1824><audio_3914><audio_2470><audio_631><audio_304><audio_4922><audio_4922><audio_2470><audio_169><audio_1><audio_1496><audio_2540><audio_4302><audio_215><audio_272><audio_4734><audio_4734><audio_1236><audio_496><audio_4><audio_3028><audio_2370><audio_1525><audio_919><audio_915><audio_2068><audio_1966><audio_3279><audio_524><audio_866><audio_1258><audio_4638><audio_2179><audio_751><audio_751><audio_1056><audio_1069><audio_2742><audio_751><audio_866><audio_2639><audio_1593><audio_1056><audio_830><audio_671><audio_4934><audio_4516><audio_5030><audio_830><audio_866><audio_1097><audio_1048><audio_3172><audio_433><audio_512><audio_1097><audio_1595><audio_3409><audio_256><audio_9><audio_1097><audio_1097><audio_2809><audio_570><audio_793><audio_2626><audio_3690><audio_3277><audio_828><audio_384><audio_1778><audio_1778><audio_1879><audio_631><audio_476><audio_4575><audio_2470><audio_2470><audio_631><audio_996><audio_2470><audio_2470><audio_2470><audio_226><audio_497><audio_2470><audio_2470><audio_2470><audio_996><audio_226><audio_2470><audio_2470><audio_2470><audio_226><audio_497><audio_2470><audio_2470><audio_2470><audio_996><audio_226><audio_2470><audio_2470><audio_2470><audio_226><audio_497><audio_2470><audio_2470><audio_2470>"


def run_atqta_infer(omni_model, tokenizer, device):
    """
    运行 ATQTA multi-turn 推理测试，验证模型正确性。

    Args:
        omni_model: OmniQwen3VLMedusaModel 实例
        tokenizer: 扩展 tokenizer
        device: 推理设备
    """
    dprint("\n=== ATQTA Multi-turn Inference Test ===")
    messages = [
        {"role": "user", "content": "", "content_audio": ATQTA_TEST_PROMPT},
    ]
    inputs = build_chat_prompt(
        tokenizer=tokenizer,
        task_type="atqta",
        messages=messages,
        add_generation_prompt=True,
        device=device,
    )
    dprint("ATQTA Prompt IDs shape:", inputs.input_ids.shape)

    generate_output = omni_model.generate(
        prompt_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=1024,
        do_sample=False,
    )

    dprint("\nGenerated Tokens:", generate_output["generated_tokens"])
    dprint(
        "\nGenerated Text:",
        tokenizer.decode(generate_output["generated_tokens"][0], skip_special_tokens=True),
    )
    dprint("=== ATQTA Test Done ===\n")


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
    # Step 3: 准备校准数据 (纯文本)
    # =================================================
    dprint("[Step 3] Preparing calibration data...")
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
