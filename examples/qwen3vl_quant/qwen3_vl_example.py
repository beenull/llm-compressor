import torch
torch.__future__.set_swap_module_params_on_conversion(True)

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from llmcompressor import Oneshot
from llmcompressor.modifiers.transform import SpinQuantModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.utils import dispatch_for_generation
from llmcompressor.modifiers.transform.spinquant import mappings, norm_mappings

# 数据集加载依赖
from datasets import concatenate_datasets, load_dataset, load_from_disk
from io import BytesIO
import base64
from PIL import Image

# Load model.
MODEL_ID = "/workspace/gaoy25@xiaopeng.com/model/qwen3_vl/Qwen3-VL-2B-Instruct/"
model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float32,attn_implementation="eager" )
processor = AutoProcessor.from_pretrained(MODEL_ID)
tokenizer = processor.tokenizer

# Select calibration dataset.
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 2048
HF_DATASETS_CACHE = "/dataset/workspace/zhangl98/hf_cache/"

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

# #带vit的
# mappings.SPINQUANT_MAPPING_REGISTRY["Qwen3VLForConditionalGeneration"] = (
#     mappings.SpinQuantMapping(
#         mm_proj=[
#             r"re:.*visual\.merger.*linear_fc2$",
#             r"re:.*visual\.deepstack_merger_list.*linear_fc2$",
#         ],
#         embedding="re:.*embed_tokens$",
#         attn="re:.*self_attn$",
#         attn_q="re:.*language_model.*q_proj$",
#         attn_k="re:.*language_model.*k_proj$",
#         attn_v="re:.*language_model.*v_proj$",
#         attn_o="re:.*language_model.*o_proj$",
#         mlp_in=[
#             r"re:.*language_model.*mlp\.up_proj$",
#             r"re:.*language_model.*mlp\.gate_proj$",
#             r"re:.*visual\.blocks.*mlp\.linear_fc1$",
#             r"re:.*visual\.deepstack_merger_list.*linear_fc1$",
#         ],
#         mlp_out=[
#             r"re:.*language_model.*mlp\.down_proj$",
#             r"re:.*visual\.blocks.*mlp\.linear_fc2$",
#             r"re:.*visual\.deepstack_merger_list.*linear_fc2$",
#         ],
#         lm_head="lm_head",
#     )
# )


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


# Get aligned calibration dataset.

def encode_base64_img(img) -> str:
    with BytesIO() as buffer:
        img.save(buffer, format="PNG")
        data = buffer.getvalue()
    return base64.b64encode(data).decode("utf-8")


def format_as_messages(example):
    role_map = {
        "human": "user",
        "gpt": "assistant",
    }
    """Format single example into messages format for TRL."""
    # example["image"] is PIL image, convert it to base64
    messages = []
    for conversation in example["conversations"]:
        message = {
            "role": role_map[conversation["from"]],
            "content": [],
        }
        content = conversation["value"]
        if "<image>" in content:
            parts = content.split("<image>")
            for i, part in enumerate(parts):
                part = part.strip()
                if i < len(parts) - 1:
                    message["content"].append(
                        {
                            "type": "image",
                            "image": f"data:image;base64,{encode_base64_img(example['image'])}",
                            "audio": None,
                        }
                    )
                if part:
                    message["content"].append({"type": "text", "text": part})
        else:
            message["content"].append({"type": "text", "text": content})
        messages.append(message)

    return {
        "messages": messages,
    }


# =================================================
# Step 3: 准备校准数据 (支持 text/vit 多模态)
# =================================================

print("[Step 3] Preparing calibration data...")
ds = load_dataset(
    "hkust-nlp/deita-6k-v0",
    split=f"train[:{NUM_CALIBRATION_SAMPLES}]",
    cache_dir=HF_DATASETS_CACHE
)
ds = ds.shuffle(seed=42)
ds = ds.map(format_as_messages,remove_columns=ds.column_names,)

def extract_text_from_content(content):
    """从 Qwen2-VL/Qwen3-VL 的多模态 content 格式中提取纯文本"""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))
        return "\n".join(texts)
    else:
        return str(content)

def preprocess(example):
    messages = example["messages"]
    # 只取首轮对话（第一条 user + 第一条 assistant）
    first_turn = messages[:2]
    # 转换格式
    converted_messages = [
        {
            "role": msg["role"],
            "content": extract_text_from_content(msg["content"])
        }
        for msg in first_turn
    ]
    # 应用 chat template，并添加 assistant 结束标记
    text = tokenizer.apply_chat_template(
        converted_messages,
        tokenize=False,
        add_generation_prompt=False,  # 不添加生成提示，因为这是完整对话
    )
    # 确保以  结尾（如果 template 没有自动添加）
    if not text.rstrip().endswith("  "):
        text = text.rstrip() + "  "
    return {"text": text}


ds = ds.map(preprocess)

def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
    )

ds = ds.map(tokenize, remove_columns=ds.column_names)

recipe_ = [
    SpinQuantModifier(
        backe_mean=True,
        learnable=False,
        rotations=["R1", "R2", "R4"],
        transform_block_size_R1=2048,
        transform_block_size_R4=256,
        transform_type="hadamard",
    ),
    # SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(
        ignore=["re:.*lm_head", "re:.*visual.*"],
        actorder=None,
        config_groups={
            "group_0": {
                "weights": {
                    "observer": "mse",
                    "observer_kwargs": {
                        "maxshrink": 0.1,
                        "patience": 10,
                        "averaging_constant": 0.05,
                        "grid": 128.0,
                        "norm": 2.0,
                    },
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "channel",
                },
                "targets": [
                    "re:.*q_proj$",
                    "re:.*k_proj$",
                    "re:.*v_proj$",
                    "re:.*o_proj$",
                    "re:.*up_proj$",
                    "re:.*gate_proj$",
                    "re:.*down_proj$",
                ],
            }
        }
    )
]
# Apply algorithms.
# tokenizer = processor.tokenizer
# 1. 先用原始模型生成一次
print("\n========== [BEFORE QUANT] SAMPLE GENERATION ==============")
dispatch_for_generation(model) #去掉
input_text = "我是小学生，给我写个100字的小作文，要符合小学生的水平"  # 可替换为更复杂的多模态输入
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
output_before = model.generate(input_ids, max_new_tokens=128, do_sample=False)
text_before = tokenizer.decode(output_before[0])
print("[原始模型输出]:\n", text_before)
print("========================================================\n")

# 2. 量化模型
# 补充 config 关键字段，兼容 llmcompressor
text_config = model.config.text_config
model.config.head_dim = text_config.head_dim
model.config.hidden_size = text_config.hidden_size
model.config.num_attention_heads = text_config.num_attention_heads
model.config.num_key_value_heads = text_config.num_key_value_heads

oneshot = Oneshot(
    model=model,
    dataset=ds,
    recipe=recipe_,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    shuffle_calibration_samples=False,
    pipeline="sequential",  # basic datafree sequential
)
oneshot()

# 3. 用量化后模型生成同样输入
print("\n========== [AFTER QUANT] SAMPLE GENERATION ==============")
dispatch_for_generation(model)
output_after = model.generate(input_ids, max_new_tokens=128, do_sample=False)
text_after = tokenizer.decode(output_after[0])
print("[量化模型输出]:\n", text_after)
print("========================================================\n")

# 4. 简单对比
print("\n[对比结果]\n原始输出:\n", text_before, "\n\n量化输出:\n", text_after)
# 折叠旋转等变换参数到权重（SpinQuant/QuaRot等）
try:
    recipe_[0]._fold_transforms_into_weights(model)
    print("[Info] 已完成 _fold_transforms_into_weights: 旋转等变换已折叠进权重")
except Exception as e:
    print(f"[Warning] _fold_transforms_into_weights 调用失败: {e}")

# # Save to disk in compressed-tensors format.
# SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "tmp"
# model.save_pretrained(SAVE_DIR, save_compressed=True)
# processor.save_pretrained(SAVE_DIR)
