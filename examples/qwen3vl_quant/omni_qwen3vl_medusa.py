"""
Omni Qwen3_VL Medusa 模型模块。

本模块实现了基于 Qwen3_VL 的 Omni 多模态模型，支持 Medusa 多头推测解码加速。
直接使用 Qwen3VLForConditionalGeneration，无需额外的 tokenizer 编解码层。
主要用于 TTS（文本转语音）、AQTA（语音转文本）以及多模态对话等任务。

从 xllm-evaluation/xllm_eval/models/omni_qwen3vl_medusa.py 拷贝。
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger
from torch import nn
from transformers import Qwen3VLForConditionalGeneration


class ResBlock(nn.Module):
    """
    残差块模块。

    用于 Medusa 头的特征变换，采用残差连接 + SiLU 激活的结构。
    权重初始化为零，使得初始状态下输出等于输入（恒等映射）。
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.linear(x))


@dataclass
class MedusaConfig:
    """Medusa 头配置。"""
    num_heads: int = 4
    num_layers: int = 1


class OmniQwen3VLMedusaModel(nn.Module):
    """
    基于 Qwen3_VL 的 Omni Medusa 模型。

    该模型使用 Qwen3VLForConditionalGeneration 作为骨干网络，结合 Medusa 多头推测解码，
    支持音频语言模型的训练和推理。
    """

    def __init__(
        self,
        language_model: Qwen3VLForConditionalGeneration,
        medusa_cfg: MedusaConfig,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.language_model = language_model
        self.medusa_cfg = medusa_cfg
        self.vocab_size = language_model.config.text_config.vocab_size
        self.hidden_size = language_model.config.text_config.hidden_size

        self.medusa_head = nn.ModuleList(
            [
                nn.Sequential(
                    *[ResBlock(self.hidden_size) for _ in range(medusa_cfg.num_layers)],
                    nn.Linear(self.hidden_size, self.vocab_size, bias=False),
                )
                for _ in range(medusa_cfg.num_heads)
            ]
        )

        model_device = self.language_model.language_model.embed_tokens.weight.device
        self.medusa_head = self.medusa_head.to(device=model_device, dtype=dtype)

    def _get_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.language_model.lm_head(hidden_states)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[Tuple] = None,
    ) -> Dict[str, Any]:
        outputs = self.language_model.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state
        logits = self._get_logits(hidden_states)
        result = {
            "hidden_states": hidden_states,
            "logits": logits,
            "past_key_values": outputs.past_key_values,
        }

        if self.medusa_cfg.num_heads > 0:
            medusa_logits = [head(hidden_states) for head in self.medusa_head]
            result["medusa_logits"] = torch.stack(medusa_logits, dim=0)
        return result

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 1024,
        eos_token_id: Optional[List[int]] = [151643, 151644, 151645, 151693],
        use_medusa: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if use_medusa:
            return self._generate_medusa(
                prompt_ids=prompt_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
        return self._generate_hf_native(
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )

    def _generate_medusa(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: Optional[List[int]] = [151643, 151644, 151645, 151693],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
    ) -> Dict[str, Any]:
        device = self.language_model.language_model.embed_tokens.weight.device
        prompt_ids = prompt_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        outputs = self.language_model.language_model(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        kv_cache = outputs.past_key_values
        generated_tokens: List[torch.Tensor] = []
        generated_logits: List[torch.Tensor] = []

        end_token_ids = eos_token_id
        end_token_ids_tensor = torch.as_tensor(end_token_ids, device=device, dtype=torch.long)

        n_step = max(1, max_new_tokens // (self.medusa_cfg.num_heads + 1))
        print("[INFO] Starting Medusa generation with max_new_tokens={}, medusa_heads={}, n_step={}".format(
            max_new_tokens, self.medusa_cfg.num_heads, n_step
        ))
        for _ in range(n_step):
            last_hidden = outputs.last_hidden_state[:, -1, :]
            logits = self._get_logits(last_hidden).unsqueeze(1)
            if self.medusa_cfg.num_heads > 0:
                medusa_logits = [head(last_hidden) for head in self.medusa_head]
                medusa_logits = torch.stack(medusa_logits, dim=1)
                last_logits = torch.cat([logits, medusa_logits], dim=1)
            else:
                last_logits = logits

            next_token = last_logits.argmax(dim=-1)
            next_token_ids = next_token
            generated_tokens.append(next_token)
            generated_logits.append(last_logits)

            if end_token_ids is not None:
                step_end = (next_token_ids[..., None] == end_token_ids_tensor).any(dim=-1).any(dim=-1)
                if step_end.any():
                    break

            outputs = self.language_model.language_model(
                input_ids=next_token_ids,
                attention_mask=None,
                past_key_values=kv_cache,
                use_cache=True,
                return_dict=True,
            )
            kv_cache = outputs.past_key_values

        return {
            "generated_tokens": torch.cat(generated_tokens, dim=-1) if generated_tokens else None,
            "generated_logits": generated_logits,
            "past_key_values": kv_cache,
        }

    def _generate_hf_native(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: Optional[List[int]] = [151643, 151644, 151645, 151693],
        pad_token_id: Optional[int] = 151643,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        device = self.language_model.language_model.embed_tokens.weight.device
        prompt_ids = prompt_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        prompt_len = prompt_ids.shape[1]

        do_sample = kwargs.pop("do_sample", True)
        temperature = kwargs.pop("temperature", 0.7)
        top_p = kwargs.pop("top_p", 0.9)
        top_k = kwargs.pop("top_k", 50)
        repetition_penalty = 1.1
        return_dict_in_generate = kwargs.pop("return_dict_in_generate", True)
        output_scores = kwargs.pop("output_scores", False)

        outputs = self.language_model.generate(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            use_cache=True,
            past_key_values=past_key_values,
            return_dict_in_generate=return_dict_in_generate,
            output_scores=output_scores,
            **kwargs,
        )

        if hasattr(outputs, "sequences"):
            sequences = outputs.sequences
            generated_logits = outputs.scores if hasattr(outputs, "scores") else None
            new_past_key_values = (
                outputs.past_key_values if hasattr(outputs, "past_key_values") else None
            )
        else:
            sequences = outputs
            generated_logits = None
            new_past_key_values = None

        generated_tokens = sequences[:, prompt_len:]
        return {
            "generated_tokens": generated_tokens,
            "generated_logits": generated_logits,
            "past_key_values": new_past_key_values,
        }

    def load_checkpoint(self, checkpoint_path: str, strict: bool = False):
        state = _load_checkpoint_state(checkpoint_path)
        state = self._remap_checkpoint_keys(state)
        return self.load_state_dict(state, strict=strict)

    @staticmethod
    def _remap_checkpoint_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        new_state: Dict[str, torch.Tensor] = {}
        for key, value in state.items():
            if key.startswith("language_model.visual.") or key.startswith("vision_encoder"):
                continue
            if key.startswith("language_model.model."):
                new_key = "language_model.model.language_model." + key[len("language_model.model."):]
            else:
                new_key = key
            new_state[new_key] = value
        return new_state

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        omni_model_name: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
    ) -> "OmniQwen3VLMedusaModel":
        if not omni_model_name:
            raise ValueError("omni_model_name must be provided for Qwen3_VL initialization.")

        state = _load_checkpoint_state(checkpoint_path)
        state = cls._remap_checkpoint_keys(state)

        lm_head_weight = state.get("language_model.lm_head.weight")
        if lm_head_weight is None:
            raise ValueError("Cannot find language_model.lm_head.weight in checkpoint")
        vocab_size, hidden_size = lm_head_weight.shape

        medusa_head_ids = sorted(
            {int(k.split(".")[1]) for k in state.keys() if k.startswith("medusa_head.")}
        )
        num_heads = (max(medusa_head_ids) + 1) if medusa_head_ids else 0

        layer_ids = sorted(
            {int(k.split(".")[2]) for k in state.keys() if k.startswith("medusa_head.0.")}
        )
        seq_len = (max(layer_ids) + 1) if layer_ids else 0
        num_medusa_layers = max(seq_len - 1, 0)

        print(
            f"[INFO] Inferred config: vocab_size={vocab_size}, hidden_size={hidden_size}, "
            f"medusa_heads={num_heads}, medusa_layers={num_medusa_layers}"
        )

        omni_model = Qwen3VLForConditionalGeneration.from_pretrained(
            omni_model_name,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

        if omni_model.config.text_config.vocab_size != vocab_size:
            print(
                "[INFO] Resizing Qwen3_VL vocab "
                f"from {omni_model.config.text_config.vocab_size} to {vocab_size} to match checkpoint."
            )
            omni_model.resize_token_embeddings(vocab_size)

        model = cls(
            language_model=omni_model,
            medusa_cfg=MedusaConfig(num_heads=num_heads, num_layers=num_medusa_layers),
            dtype=dtype,
        )

        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[WARN] missing keys: {missing}")
        if unexpected:
            print(f"[WARN] unexpected keys: {unexpected}")
        return model


def _load_checkpoint_state(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    load_kwargs: Dict[str, Any] = {"map_location": "cpu"}
    try:
        ckpt = torch.load(checkpoint_path, weights_only=True, mmap=True, **load_kwargs)
    except TypeError:
        try:
            ckpt = torch.load(checkpoint_path, weights_only=True, **load_kwargs)
        except TypeError:
            ckpt = torch.load(checkpoint_path, **load_kwargs)
    return ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
