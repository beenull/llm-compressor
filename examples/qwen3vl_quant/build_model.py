"""
Omni 模型构建模块。

提供构建 OmniQwen3VLMedusaModel 实例和 tokenizer 的函数。
从 xllm-evaluation/xllm_eval/build_model.py 拷贝。
"""

import torch
from transformers import AutoTokenizer

from omni_qwen3vl_medusa import OmniQwen3VLMedusaModel


def build_model(
    omni_model_name: str,
    omni_model_tokenizer_path: str,
    omni_model_ckpt_path: str,
    device: str = "cuda:0",
):
    """
    构建 OmniQwen3VLMedusaModel 实例和tokenizer.

    Args:
        omni_model_name: 初始化 HF Omni model 名称或路径。
        omni_model_tokenizer_path: Omni model tokenizer 路径（独立于 model_name）。
        omni_model_ckpt_path: Omni model checkpoint 路径。
        device: auto/cpu/cuda:0。
    """

    omni_model_tokenizer = AutoTokenizer.from_pretrained(
        omni_model_tokenizer_path, trust_remote_code=True, local_files_only=True
    )

    omni_model = OmniQwen3VLMedusaModel.from_checkpoint(
        checkpoint_path=omni_model_ckpt_path,
        omni_model_name=omni_model_name,
        dtype=torch.bfloat16,
        device_map=device,
    )

    return omni_model, omni_model_tokenizer
