from datasets import concatenate_datasets, load_dataset, load_from_disk
from io import BytesIO
import base64
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


enable_modality = {
    # "vit",
    "text"
}

HF_DATASETS_CACHE = "/dataset/workspace/zhangl98/hf_cache/"

ds_vl = load_dataset(
    "lmms-lab/LLaVA-OneVision-Data", 
    "FigureQA(MathV360K)", 
    split="train[:128]",
    cache_dir= HF_DATASETS_CACHE
)


ds_text = load_dataset("hkust-nlp/deita-6k-v0", 
                       split="train[:128]",
                       cache_dir= HF_DATASETS_CACHE)


ds_wiki = load_from_disk("/dataset/workspace/zhangl98/dataset/calib/wikitext2/")


ds_vl = ds_vl.map(
    format_as_messages,
    remove_columns=ds_vl.column_names,
    # num_proc=6,
    # fn_kwargs={"prompt": "What does the image show?"},
)

ds_text = ds_text.map(
    format_as_messages,
    remove_columns=ds_text.column_names,
    # num_proc=6,
    # fn_kwargs={"prompt": "Please transcribe the audio."},
)

ds = []
if "vit" in enable_modality:
    ds.append(ds_vl)
if "text" in enable_modality:
    ds.append(ds_text)
ds = concatenate_datasets(ds)
ds = ds.shuffle(seed=42)

