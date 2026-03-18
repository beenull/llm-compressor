"""
Qwen3-VL 2B: OSTQuant Training (Phase 1)

可学习旋转 R1+R2 + STE fake-quantization + KL 蒸馏训练。
使用 FSDP 分布式训练。训练完成后保存 transformed model，
用于后续 Phase 2 (R4 + GPTQ) 量化。

Usage:
    torchrun --nproc_per_node=2 examples/qwen3vl_quant/ostquant_train.py \
        --config examples/qwen3vl_quant/configs/train.yaml
"""

import argparse
import contextlib
import copy
import datetime
import os
from collections import OrderedDict

import torch
import torch.distributed as dist
import yaml
from accelerate import init_empty_weights
from compressed_tensors.quantization import enable_quantization
from compressed_tensors.transform.factory.base import TransformBase
from datasets import load_dataset
from easydict import EasyDict
from loguru import logger
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as PT_FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

from llmcompressor.core.state import State
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform import SpinQuantModifier
from llmcompressor.modifiers.transform.spinquant import mappings, norm_mappings
from llmcompressor.train.fsdp_trainer import MyTrainer
from llmcompressor.train.train_utils import LLMCTrainingArguments, TeacherModel
from llmcompressor.utils import helpers
from llmcompressor.utils.pytorch.module import (
    build_weight_tied_map_with_unionfind,
    patch_module_non_persistent_buffers,
)

torch.fx.experimental._config.meta_nonzero_assume_all_nonzero = True

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


#################### configurations ####################
MODEL_ID = "/dataset/workspace/models/Qwen3-VL-2B-Instruct"  # 替换为实际路径
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 2048
model_dtype = torch.bfloat16
enable_modality = {"text"}  # 仅量化 language model
#################### configurations ####################

pretrain = "origin"
flag = "ostquant"
flag += str(tuple(enable_modality)).replace("'", "").replace(",", "|")

SAVE_DIR = (
    "/tmp/" + MODEL_ID.rstrip("/").split("/")[-1] + f"-{pretrain}-{flag}" + "-trans"
)


# =====================================================
# 数据准备
# =====================================================
ds_text = load_dataset("hkust-nlp/deita-6k-v0", split="train[:256]")


def format_as_messages(example):
    role_map = {"human": "user", "gpt": "assistant"}
    messages = []
    for conversation in example["conversations"]:
        message = {
            "role": role_map[conversation["from"]],
            "content": [{"type": "text", "text": conversation["value"]}],
        }
        messages.append(message)
    return {"messages": messages}


ds = ds_text.map(format_as_messages, remove_columns=ds_text.column_names)
ds = ds.shuffle(seed=42)


# =====================================================
# SpinQuant R1+R2 可学习旋转初始化
# =====================================================
@torch.no_grad()
def pre_compression_spinquant(model):
    state = State()
    state.update(model=model)
    recipe_ = [
        SpinQuantModifier(
            do_fold=False,
            backe_mean=False,
            learnable=True,
            rotations=["R1", "R2"],
            transform_block_size_R1=1536,  # Qwen3-VL 2B hidden_size
            transform_type="random-hadamard",
            sequential_onload=not RANK_OTHER,
        )
    ]
    with contextlib.ExitStack():
        for mod in recipe_:
            mod.on_initialize(state=state)
        recipe_[0].on_start(state=state, event=None)

    return state, recipe_, model


# =====================================================
# Fake-quantization 初始化 (STE)
# =====================================================
@torch.no_grad()
def pre_compression_fakequant(model):
    state = State()
    state.update(model=model)
    recipe_ = [
        QuantizationModifier(
            ignore=["re:.*lm_head", "re:.*visual.*"],
            config_groups={
                "group_0": {
                    "weights": {
                        "observer": "minmax",
                        "num_bits": 4,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "channel",
                        "dynamic": True,
                    },
                    "input_activations": {
                        "observer": "minmax",
                        "num_bits": 8,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "tensor",
                        "dynamic": True,
                    },
                    "targets": [
                        r"re:.*up_proj$",
                        r"re:.*gate_proj$",
                        r"re:.*q_proj$",
                        r"re:.*k_proj$",
                        r"re:.*v_proj$",
                        r"re:.*o_proj$",
                    ],
                    "ste": True,
                },
                "group_1": {
                    "weights": {
                        "observer": "minmax",
                        "num_bits": 4,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "channel",
                        "dynamic": True,
                    },
                    "input_activations": {
                        "observer": "minmax",
                        "num_bits": 16,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "tensor",
                        "dynamic": True,
                    },
                    "targets": [r"re:.*down_proj$"],
                    "ste": True,
                },
            },
        )
    ]

    with contextlib.ExitStack():
        with torch.nn.utils.parametrize.cached():
            for mod in recipe_:
                mod.on_initialize(state=state)
            model.apply(enable_quantization)

    return state, recipe_, model


# =====================================================
# Data Collator
# =====================================================
class DataCollatorForQwen3VL(DataCollatorForLanguageModeling):
    def __init__(self, processor):
        self.processor = processor
        self.assistant_start_tokens = processor(text=["<|im_start|>assistant\n"])[
            "input_ids"
        ][0]
        self.assistant_end_tokens = processor(text=["<|im_end|>\n"])["input_ids"][0]

    def __call__(self, examples):
        conversations = [
            [
                {
                    "role": turn["role"],
                    "content": [
                        {k: v for k, v in content.items() if v is not None}
                        for content in turn["content"]
                    ],
                }
                for turn in example["messages"]
            ]
            for example in examples
        ]
        text = self.processor.apply_chat_template(
            conversations, add_generation_prompt=False, tokenize=False
        )
        batch = self.processor(
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
        )

        labels = (
            torch.ones_like(batch["input_ids"], dtype=batch["input_ids"].dtype) * -100
        )

        for batch_idx, token_ids in enumerate(batch["input_ids"].tolist()):
            valid = False
            pos = 0
            is_assistant_response = False
            while pos < len(token_ids):
                if is_assistant_response:
                    valid = True
                    if token_ids[pos] == self.assistant_end_tokens[0]:
                        is_assistant_end = True
                        for i, t in enumerate(self.assistant_end_tokens[1:], 1):
                            if pos + i >= len(token_ids) or token_ids[pos + i] != t:
                                is_assistant_end = False
                                break
                        if is_assistant_end:
                            is_assistant_response = False
                            for i in range(pos, pos + len(self.assistant_end_tokens)):
                                labels[batch_idx, i] = token_ids[i]
                            pos += len(self.assistant_end_tokens)
                        else:
                            labels[batch_idx, pos] = token_ids[pos]
                            pos += 1
                    else:
                        labels[batch_idx, pos] = token_ids[pos]
                        pos += 1
                else:
                    if token_ids[pos] == self.assistant_start_tokens[0]:
                        is_assistant_start = True
                        for i, t in enumerate(self.assistant_start_tokens[1:], 1):
                            if pos + i >= len(token_ids) or token_ids[pos + i] != t:
                                is_assistant_start = False
                                break
                        if is_assistant_start:
                            is_assistant_response = True
                            pos += len(self.assistant_start_tokens)
                        else:
                            pos += 1
                    else:
                        pos += 1
            if not valid:
                labels[batch_idx, :] = token_ids

        batch["labels"] = labels
        return batch


# =====================================================
# 训练后保存 transformed model
# =====================================================
@torch.no_grad()
def post_compression(model, processor, recipe_):
    import re

    from compressed_tensors.utils.match import match_named_modules

    _h = set()
    transform_state_dict = OrderedDict()

    for name, module in model.named_modules():
        if isinstance(module, TransformBase):
            if module in _h or id(module.scheme) in _h:
                continue
            _h.add((module if module.scheme.block_wise else id(module.scheme)))
            print(f"{name}: {module}")
            transform_state_dict.update({name: module.state_dict()})

    to_removes = []
    for name, module in model.named_modules():
        for child_name, child_module in module.named_children():
            if isinstance(child_module, TransformBase):
                to_removes.append((module, child_name))
    for module, child_name in to_removes:
        delattr(module, child_name)

    # 移除 fake-quant 状态
    quantized_name_set = set()
    for _, module in match_named_modules(
        model, recipe_[-1].resolved_targets, recipe_[-1].ignore
    ):
        if hasattr(module, "quantization_status"):
            quantized_name_set.add(re.sub(r"\d+", "X", _))
            delattr(module, "quantization_status")
            delattr(module, "quantization_enabled")
            delattr(module, "quantization_scheme")
            for key in list(module._parameters.keys()):
                if key.endswith("_scale") or key.endswith("_zero_point"):
                    delattr(module, key)
    print(f"Total quantized modules: {quantized_name_set}")

    model.save_pretrained(SAVE_DIR)
    processor.save_pretrained(SAVE_DIR)
    torch.save(transform_state_dict, f"{SAVE_DIR}/transform_state_dict.pt")
    print(f"Transformed model saved to {SAVE_DIR}")


# =====================================================
# 分布式加载
# =====================================================
def dist_load_model(model_path=MODEL_ID, load_processor=False):
    processor = None
    if RANK_OTHER:
        with init_empty_weights():
            model = Qwen3VLForConditionalGeneration._from_config(
                model_config, dtype=model_dtype
            )
    else:
        if load_processor:
            processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True
            )
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=model_dtype
        )
    return model, processor


# =====================================================
# FSDP 训练主流程
# =====================================================
def fsdp_main(model, config):
    training_params_cnt = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            training_params_cnt += 1
    weight_tied_name_map = build_weight_tied_map_with_unionfind(model)

    state_dict = model.state_dict()
    model.to("meta")
    model_to_train = copy.deepcopy(model)
    if not RANK_OTHER:
        model_to_train.load_state_dict(state_dict, assign=True)
    del state_dict

    train_processor = AutoProcessor.from_pretrained(
        pretrained_model_name_or_path=MODEL_ID,
        model_max_length=MAX_SEQUENCE_LENGTH,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
    )
    train_args = LLMCTrainingArguments(**config.train_args)
    need_teacher = train_args.special.get("loss_type", "origin") not in (
        "origin",
        "DFT",
    )
    model_to_train.train()

    if need_teacher:
        model_path = train_args.special.get("teacher_path", MODEL_ID)
        teacher_model, _ = dist_load_model(model_path)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        teacher_model.config.text_config.use_cache = False
        model_to_train.teacher = TeacherModel(teacher_model)

    model_to_train.config.text_config.use_cache = False
    assert len(set(weight_tied_name_map.values())) == training_params_cnt
    trainer = MyTrainer(
        model=model_to_train,
        args=train_args,
        train_dataset=ds,
        eval_dataset=None,
        data_collator=DataCollatorForQwen3VL(train_processor),
        weight_tied_name_map=weight_tied_name_map,
        ignored_modules=[],
    )
    trainer.train()
    dist.barrier()
    if hasattr(trainer.model, "_orig_mod"):
        unwrapped_model = trainer.model._orig_mod
    else:
        unwrapped_model = trainer.model
    state_dict = trainer.accelerator.get_state_dict(unwrapped_model)
    if not RANK_OTHER:
        state_dict = {
            k: v for k, v in state_dict.items() if not k.startswith("teacher")
        }
        model.load_state_dict(state_dict, assign=True)
        trainer.register_tied_parameters(model, weight_tied_name_map)


def cleanup():
    dist.barrier()
    dist.destroy_process_group()


def setup():
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=7200))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
        config = EasyDict(config)

    setup()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    RANK_OTHER = dist.is_initialized() and dist.get_rank() != 0
    model_config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    if RANK_OTHER:
        logger.remove()

    model, processor = dist_load_model(load_processor=True)
    with patch_module_non_persistent_buffers(model):
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        # Step 1: 初始化 R1+R2 可学习旋转
        state_text, recipe_text, model = pre_compression_spinquant(model)

        # Step 2: 初始化 STE fake-quantization
        state, recipe_, model = pre_compression_fakequant(model)

        dist.barrier()

        # Step 3: FSDP 训练 (KL 蒸馏 + STE)
        fsdp_main(model, config)

    torch.cuda.empty_cache()
    dist.barrier()

    if not RANK_OTHER:
        # Step 4: fold 旋转到权重并保存
        recipe_text[0]._fold_transforms_into_weights(state_text.model)
        post_compression(model, processor, recipe_)

    cleanup()
