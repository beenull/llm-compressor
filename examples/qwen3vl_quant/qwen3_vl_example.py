from auto_round.calib_dataset import get_dataset
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.autoround import AutoRoundModifier
from llmcompressor.utils import dispatch_for_generation

# Load model.
MODEL_ID = "/workspace/gaoy25@xiaopeng.com/model/qwen3_vl/Qwen3-VL-2B-Instruct/"
model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype="auto")
processor = AutoProcessor.from_pretrained(MODEL_ID)
tokenizer = processor.tokenizer


import auto_round.autoround as ar_module
from auto_round.compressors.mllm.compressor import MLLMCompressor

# 保存原始 AutoRound
_OriginalAutoRound = ar_module.AutoRound

def patched_autoround_new(cls, *args, **kwargs):
    # 强制注入 processor
    if 'processor' not in kwargs or kwargs['processor'] is None:
        kwargs['processor'] = processor
        print(f"[Patch] Injected processor: {kwargs['processor']}")
    
    # 修复空 tokenizer
    if kwargs.get('tokenizer') == '':
        kwargs['tokenizer'] = processor.tokenizer
    
    return _OriginalAutoRound(*args, **kwargs)

ar_module.AutoRound = patched_autoround_new
ar_module.AutoRound.__new__ = staticmethod(patched_autoround_new)



# Select calibration dataset.
NUM_CALIBRATION_SAMPLES = 128
MAX_SEQUENCE_LENGTH = 2048
# Get aligned calibration dataset.
# 数据集加载依赖
from datasets import concatenate_datasets, load_dataset, load_from_disk
from io import BytesIO
import base64
from PIL import Image

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
HF_DATASETS_CACHE = "/dataset/workspace/zhangl98/hf_cache/"
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

# def tokenize(sample):
#     return tokenizer(
#         sample["text"],
#         padding=False,
#         max_length=MAX_SEQUENCE_LENGTH,
#         truncation=True,
#     )

# ds = ds.map(tokenize, remove_columns=ds.column_names)

# ds = get_dataset(
#     tokenizer=tokenizer,
#     seqlen=MAX_SEQUENCE_LENGTH,
#     nsamples=NUM_CALIBRATION_SAMPLES,
# )


# Configure the quantization algorithm to run.
#   * quantize the weights to 4 bit with AutoRound with a group size 128
recipe = AutoRoundModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=["re:.*lm_head", "re:.*visual.*"],
    iters=200,
)
# recipe = "examples/qwen3vl_quant/configs/quarot.yaml"
# Apply algorithms.
# tokenizer = processor.tokenizer
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    processor=processor,
    # tokenizer = processor.tokenizer,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    # disable shuffling to get slightly better mmlu score
    shuffle_calibration_samples=False,
)

print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
    model.device
)
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")


# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4-AutoRound"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)
