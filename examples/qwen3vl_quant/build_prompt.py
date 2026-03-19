"""
Prompt 构建模块。

提供多任务 prompt 构建函数 (AQTA, ATQTA, TTS, ASR 等)。
从 xllm-evaluation/xllm_eval/build_prompt.py 拷贝。
"""

import os

import torch
from loguru import logger

# 非 DEBUG 模式下屏蔽 loguru INFO 日志
if os.environ.get("DEBUG", "").lower() not in ("1", "true", "yes"):
    logger.remove()
    logger.add(lambda msg: None, level="WARNING")


SYSTEM_PROMPTS = {
    "aqtaa": "你是语音助手小p。用户将通过语音与你交流，你需要先生成文本回复，然后将该文本转换为语音进行输出。请确保回复内容自然流畅，适合语音播放。",
    "tqtaa": "你是语音助手小p。用户将通过文本与你交流，你需要先生成文本回复，然后将该文本转换为语音进行输出。请确保回复内容自然流畅，适合语音播放。",
    "atqaa": "你是语音助手小p。用户将通过语音与你交流，系统会同时提供语音对应的文本转写内容。请根据用户的语音及转写文本，以纯文本形式进行回复。",
    "aqaa": "你是语音助手小p。用户将通过语音与你交流，你需要直接以语音形式进行回复。请确保回复内容简洁自然，适合语音对话场景。",
    "aqta": "你是语音助手小p。用户将通过语音与你交流，你需要以纯文本形式进行回复。请准确理解用户语音内容并给出恰当的文字回应。",
    "tqaa": "你是语音助手小p。用户将通过文本与你交流，你需要直接以语音形式进行回复。请确保回复内容简洁自然，适合语音播放。",
    "atqta": "你是语音助手小p。用户将通过文本与语音与你交流，你需要以纯文本形式进行回复。请准确理解用户文本和语音内容并给出恰当的文字回应。",
}


FORMAT_CONFIG = {
    "aqtaa": ("audio", "text+audio"),
    "tqtaa": ("text", "text+audio"),
    "atqaa": ("audio+text", "text"),
    "aqaa": ("audio", "audio"),
    "aqta": ("audio", "text"),
    "tqaa": ("text", "audio"),
    "atqta": ("text+audio", "text"),
}


def build_task_prompt(tokenizer, task_type, text_list, audio_text_list, device):
    """构造 TTS/ASR 任务 prompt_ids。"""

    if tokenizer is None:
        raise ValueError("build_task_prompt 需要在构造时传入 `tokenizer=`")

    if task_type == "tts":
        text = "".join(text_list)
        prompt = f"<|TTS|>将如下文本转换为语音：{text}<audio_start>"
    elif task_type == "asr":
        audio_token = "".join(audio_text_list)
        prompt = f"<|ASR|><audio_start>{audio_token}<audio_end>"
    elif task_type == "aqta":
        audio_token = "".join(audio_text_list)
        prompt = f"<|AQTA|><|im_start|>system\n你是语音助手小p。用户将通过语音与你交流，你需要以纯文本形式进行回复。请准确理解用户语音内容并给出恰当的文字回应。<|im_end|>\n<|im_start|>user\n<audio_start>{audio_token}<audio_end><|im_end|>\n<|im_start|>assistant\n"
    else:
        raise ValueError(f"task_type {task_type} not supported")

    logger.info(f"Constructed task prompt for {task_type}: {prompt}")

    text = prompt
    inputs = tokenizer([text], return_tensors="pt").to(device)
    inputs = inputs.to(device)

    return inputs


def build_chat_prompt(tokenizer, task_type, messages, device, add_generation_prompt=True):
    """
    按 任务模板与示例.md 多轮 AQAA 格式构建 prompt。

    messages: 列表，每项为 {"role": "user"|"assistant", "content": str (可选), "content_audio": List[int] (可选)}。
    """
    for message in messages:
        assert isinstance(message, dict) and "role" in message and "content" in message

    task_type = task_type.lower()
    user_mode, asst_mode = FORMAT_CONFIG[task_type]
    tag = f"<|{task_type.upper()}|>"
    system = SYSTEM_PROMPTS[task_type]
    lines = [tag + "<|im_start|>system", system + "<|im_end|>"]

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content") or ""
        content_audio = msg.get("content_audio") or []
        is_user = role == "user"
        mode = user_mode if is_user else asst_mode

        part = []
        if mode == "text":
            part.append(content)
        elif mode == "audio":
            part.append(content_audio)
        elif mode == "text+audio":
            part.append(content)
            part.append(content_audio)
        else:  # audio+text
            part.append(content_audio)
            part.append(content)

        line_content = "".join(part).strip()
        lines.append("<|im_start|>" + role)
        lines.append(line_content + "<|im_end|>")

    if add_generation_prompt:
        lines.append("<|im_start|>assistant")
    prompt = "\n".join(lines)
    logger.info(f"Constructed chat prompt for {task_type}:\n{prompt}")

    text = prompt
    inputs = tokenizer([text], return_tensors="pt").to(device)
    inputs = inputs.to(device)

    return inputs
