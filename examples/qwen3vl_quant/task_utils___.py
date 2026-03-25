ATQTA_TEST_PROMPT = """
<audio_667><audio_993><audio_4217><audio_1133><audio_2141><audio_73><audio_933><audio_2141><audio_2079><audio_1797><audio_265><audio_997><audio_3113><audio_2288><audio_4106><audio_331><audio_776><audio_2639><audio_2639><audio_4911><audio_109><audio_673><audio_1127><audio_4436><audio_1782><audio_799><audio_12><audio_1367><audio_1879><audio_1089><audio_60><audio_262><audio_2160><audio_3839><audio_4371><audio_262><audio_936><audio_3453><audio_1816><audio_2759><audio_262><audio_384><audio_2759><audio_3645><audio_1824><audio_631><audio_497><audio_2779><audio_1055><audio_2470><audio_631><audio_226><audio_2141><audio_2470><audio_2470><audio_226><audio_497><audio_2470><audio_2470><audio_2470><audio_169><audio_148><audio_2141><audio_2758><audio_2493><audio_756><audio_214><audio_2639><audio_3423><audio_1604><audio_273><audio_273><audio_4938><audio_4203><audio_2200><audio_908><audio_9><audio_1295><audio_2019><audio_4575><audio_26><audio_283><audio_2722><audio_1327><audio_1172><audio_18><audio_355><audio_4073><audio_4513><audio_4641><audio_36><audio_36><audio_1904><audio_1085><audio_2785><audio_53><audio_53><audio_2624><audio_4177><audio_3690><audio_53><audio_53><audio_4274><audio_3277><audio_2494><audio_866><audio_818><audio_1049><audio_3833><audio_2342><audio_803><audio_803><audio_4624><audio_3670><audio_4569><audio_384><audio_936><audio_4734><audio_4734><audio_3708><audio_431><audio_631><audio_4922><audio_2141><audio_2470><audio_226><audio_476><audio_2470><audio_2470><audio_2470><audio_935><audio_327><audio_1029><audio_3654><audio_1050><audio_327><audio_214><audio_1050><audio_3959><audio_1097><audio_898><audio_8><audio_1048><audio_1426><audio_1518><audio_8><audio_361><audio_1044><audio_4842><audio_1075><audio_660><audio_660><audio_2639><audio_2051><audio_3956><audio_866><audio_851><audio_3956><audio_1028><audio_4289><audio_905><audio_799><audio_4086><audio_1660><audio_2179><audio_36><audio_950><audio_1236><audio_4217><audio_1535><audio_965><audio_965><audio_2316><audio_1068><audio_2061><audio_866><audio_896><audio_1432><audio_3759><audio_1049><audio_607><audio_866><audio_2264><audio_2537><audio_1027><audio_423><audio_512><audio_3788><audio_1066><audio_1278><audio_423><audio_423><audio_4073><audio_2313><audio_1639><audio_384><audio_936><audio_5080><audio_3670><audio_4673><audio_631><audio_631><audio_1050><audio_2141><audio_1027><audio_118><audio_550><audio_4922><audio_4922><audio_2430><audio_169><audio_191><audio_3708><audio_3190><audio_2440><audio_191><audio_219><audio_3738><audio_1375><audio_2973><audio_231><audio_231><audio_1599><audio_1986><audio_1052><audio_231><audio_441><audio_2124><audio_1367><audio_1367><audio_85><audio_76><audio_1824><audio_3914><audio_2470><audio_631><audio_304><audio_4922><audio_4922><audio_2470><audio_169><audio_1><audio_1496><audio_2540><audio_4302><audio_215><audio_272><audio_4734><audio_4734><audio_1236><audio_496><audio_4><audio_3028><audio_2370><audio_1525><audio_919><audio_915><audio_2068><audio_1966><audio_3279><audio_524><audio_866><audio_1258><audio_4638><audio_2179><audio_751><audio_751><audio_1056><audio_1069><audio_2742><audio_751><audio_866><audio_2639><audio_1593><audio_1056><audio_830><audio_671><audio_4934><audio_4516><audio_5030><audio_830><audio_866><audio_1097><audio_1048><audio_3172><audio_433><audio_512><audio_1097><audio_1595><audio_3409><audio_256><audio_9><audio_1097><audio_1097><audio_2809><audio_570><audio_793><audio_2626><audio_3690><audio_3277><audio_828><audio_384><audio_1778><audio_1778><audio_1879><audio_631><audio_476><audio_4575><audio_2470><audio_2470><audio_631><audio_996><audio_2470><audio_2470><audio_2470><audio_226><audio_497><audio_2470><audio_2470><audio_2470><audio_996><audio_226><audio_2470><audio_2470><audio_2470><audio_226><audio_497><audio_2470><audio_2470><audio_2470><audio_996><audio_226><audio_2470><audio_2470><audio_2470><audio_226><audio_497><audio_2470><audio_2470><audio_2470>"
"""

def run_atqta_infer(omni_model, tokenizer, device):
    """
    运行 ATQTA multi-turn 推理测试，验证模型正确性。
    Args:
        omni_model: OmniQwen3VLMedusaModel 实例
        tokenizer: 扩展 tokenizer
        device: 推理设备
    """
    from .quarot_gptq import dprint, build_chat_prompt
    messages = [
        {"role": "user", "content": "", "content_audio": ATQTA_TEST_PROMPT},
    ]
    dprint("\n=== ATQTA Multi-turn Inference Test ===")
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
"""
task_utils.py

包含多模态校准数据集加载、格式化、预处理等与主流程无关的工具函数。
"""
import base64
from io import BytesIO
from datasets import concatenate_datasets, load_dataset, load_from_disk
from PIL import Image

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

def get_multimodal_calib_dataset(NUM_CALIBRATION_SAMPLES, enable_modality, HF_DATASETS_CACHE):
    """
    加载多模态校准数据集，支持 text/vit。
    返回 datasets.Dataset
    """
    ds_vl = load_dataset(
        "lmms-lab/LLaVA-OneVision-Data", 
        "FigureQA(MathV360K)", 
        split=f"train[:{NUM_CALIBRATION_SAMPLES}]",
        cache_dir=HF_DATASETS_CACHE
    )
    ds_text = load_dataset(
        "hkust-nlp/deita-6k-v0",
        split=f"train[:{NUM_CALIBRATION_SAMPLES}]",
        cache_dir=HF_DATASETS_CACHE
    )
    # 可选：加载本地 wikitext2 校准集
    # ds_wiki = load_from_disk("/dataset/workspace/zhangl98/dataset/calib/wikitext2/")
    ds_vl = ds_vl.map(
        format_as_messages,
        remove_columns=ds_vl.column_names,
    )
    ds_text = ds_text.map(
        format_as_messages,
        remove_columns=ds_text.column_names,
    )
    ds = []
    if "vit" in enable_modality:
        ds.append(ds_vl)
    if "text" in enable_modality:
        ds.append(ds_text)
    ds = concatenate_datasets(ds)
    ds = ds.shuffle(seed=42)
    return ds

def preprocess_for_chat_template(example, tokenizer):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }

def tokenize_for_calib(sample, tokenizer, max_length):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=max_length,
        truncation=True,
    )
