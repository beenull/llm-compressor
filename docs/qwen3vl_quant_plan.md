# Qwen3-VL 量化方案

> 基于 llm-compressor 框架对 Qwen3-VL 2B 进行 QuaRot+GPTQ 和 OSTQuant+GPTQ 量化的完整方案。

---

## 1. 目标

| 步骤 | 方法 | 说明 |
|---|---|---|
| Step 1 | **QuaRot + GPTQ** | 非学习旋转 (R1+R2+R4) + GPTQ W4 量化，作为 baseline |
| Step 2 | **OSTQuant + GPTQ** | 可学习旋转 (R1+R2 STE) + R4+GPTQ 后量化，更高精度 |

---

## 2. Qwen3-VL 2B 模型结构

### 2.1 模型参数

| 参数 | 值 |
|---|---|
| 模型类名 | `Qwen3VLForConditionalGeneration` |
| hidden_size | 1536 |
| num_layers | 28 |
| num_attention_heads | 12 |
| num_kv_heads | 2 |
| intermediate_size | 8960 |
| Decoder Layer 类名 | `Qwen3VLTextDecoderLayer` |
| Norm 类型 | RMSNorm |

### 2.2 模型层级路径

```
Qwen3VLForConditionalGeneration
├── model: Qwen3VLModel
│   ├── visual: Qwen3VLVisionModel         ← 量化时 ignore
│   │   ├── patch_embed / pos_embed
│   │   ├── blocks[i]: Qwen3VLVisionBlock
│   │   │   ├── norm1 / norm2 (LayerNorm)
│   │   │   ├── attn.qkv / attn.proj
│   │   │   └── mlp.linear_fc1 / linear_fc2
│   │   ├── merger
│   │   └── deepstack_merger_list          ★ Qwen3-VL 新增
│   └── language_model: Qwen3VLTextModel   ★ 保留 language_model 前缀 (同 Qwen2.5-VL)
│       ├── embed_tokens
│       ├── layers[i]: Qwen3VLTextDecoderLayer
│       │   ├── input_layernorm (RMSNorm)
│       │   ├── self_attn
│       │   │   ├── q_proj / k_proj / v_proj / o_proj
│       │   │   ├── q_norm / k_norm          ★ Qwen3-VL 新增 (RMSNorm on head_dim)
│       │   ├── post_attention_layernorm (RMSNorm)
│       │   └── mlp
│       │       ├── gate_proj / up_proj / down_proj
│       ├── norm (RMSNorm)
│       └── rotary_emb
└── lm_head
```

### 2.3 与 Qwen2.5-VL 的关键差异

| 特性 | Qwen2.5-VL | Qwen3-VL |
|---|---|---|
| 路径前缀 | `model.language_model.layers.{i}` | **完全一致** ✅ |
| Decoder Layer 类名 | `Qwen2_5_VLDecoderLayer` | `Qwen3VLTextDecoderLayer` |
| QK Norm | ❌ | ✅ `q_norm` + `k_norm` |
| Vision DeepStack | ❌ | ✅ `deepstack_merger_list` |
| SpinQuant regex | `re:.*language_model.*q_proj$` | **可复用** ✅ |

**结论**: SpinQuant/Norm 映射 regex 可直接从 Qwen2.5-VL 复用，只需更改注册 key。

---

## 3. 注册映射 (Registry Mappings)

### 3.1 SpinQuant 映射

```python
from llmcompressor.modifiers.transform.spinquant import mappings, norm_mappings

mappings.SPINQUANT_MAPPING_REGISTRY["Qwen3VLForConditionalGeneration"] = (
    mappings.SpinQuantMapping(
        mm_proj=[r"re:.*visual\.merger.*linear_fc2$"],  # Qwen3-VL merger 用 linear_fc2
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
```

### 3.2 Norm 映射

```python
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
```

---

## 4. Step 1: QuaRot + GPTQ

### 4.1 方法简介

QuaRot 是 SpinQuant 的非学习版本:
- **R1**: embedding 后插入 Hadamard 旋转 (block_size = hidden_size = 1536)
- **R2**: attention O 投影后的旋转
- **R4**: down_proj 前的 block-wise 旋转 (block_size = 256)
- 所有旋转矩阵为固定 Hadamard 矩阵，**不需要训练**，也**不需要校准数据** (datafree)

然后使用 GPTQ 做 W4 量化 (需要校准数据)。

### 4.2 执行流程

```
QuaRot (datafree)        GPTQ (需校准数据)
   R1+R2+R4 旋转   →   W4 channel symmetric 量化
   ─────────────────────────────────────────────
   单个 oneshot() 调用即可完成
```

### 4.3 配置文件

使用 `examples/qwen3vl_quant/configs/quarot_gptq.yaml`:

```yaml
quant_stage:
  quant_modifiers:
    SpinQuantModifier:
      rotations: ["R1", "R2", "R4"]
      transform_block_size_R1: 1536   # = hidden_size of Qwen3-VL 2B
      transform_block_size_R4: 256
      transform_type: "hadamard"
    GPTQModifier:
      ignore: ["re:.*lm_head", "re:.*visual.*"]
      actorder: null
      config_groups:
        group_0:
          weights:
            observer: mse
            observer_kwargs:
              maxshrink: 0.1
              patience: 10
            num_bits: 4
            type: int
            symmetric: true
            strategy: channel
          targets:
            - "re:.*q_proj$"
            - "re:.*k_proj$"
            - "re:.*v_proj$"
            - "re:.*o_proj$"
            - "re:.*up_proj$"
            - "re:.*gate_proj$"
            - "re:.*down_proj$"
```

### 4.4 脚本

`examples/qwen3vl_quant/quarot_gptq.py`

### 4.5 运行命令

```bash
python examples/qwen3vl_quant/quarot_gptq.py
```

---

## 5. Step 2: OSTQuant + GPTQ

### 5.1 方法简介

OSTQuant 是 SpinQuant 的可学习版本:
- **Phase 1 (Training)**: R1+R2 旋转矩阵通过 STE (Straight-Through Estimator) + KL 蒸馏进行训练优化
  - 需要 FSDP 分布式训练
  - 使用 fake-quantization 感知训练
  - 训练完成后保存 transformed model (含学习到的旋转矩阵)
- **Phase 2 (PostQuant)**: 在 transformed model 上加 R4 旋转 + GPTQ W4 量化

### 5.2 执行流程

```
Phase 1: OSTQuant Training (FSDP)     Phase 2: PostQuant
   R1+R2 可学习旋转                      R4 旋转 + GPTQ W4
   STE fake-quant                        oneshot() 调用
   KL 蒸馏 (with teacher)
   ────────────────────────           ─────────────────────
   torchrun 多卡训练                    单卡 python 即可
   输出: transformed model              输出: 最终量化模型
```

### 5.3 配置文件

Phase 1 训练:
- 旋转 + fake-quant: `examples/qwen3vl_quant/configs/ostquant_train.yaml`
- 训练参数: `examples/qwen3vl_quant/configs/train.yaml`

Phase 2 后量化:
- R4 + GPTQ: `examples/qwen3vl_quant/configs/r4_gptq.yaml`

### 5.4 脚本

- Phase 1: `examples/qwen3vl_quant/ostquant_train.py`
- Phase 2: `examples/qwen3vl_quant/ostquant_postquant.py`

### 5.5 运行命令

```bash
# Phase 1: OSTQuant Training (2 GPU)
torchrun --nproc_per_node=2 examples/qwen3vl_quant/ostquant_train.py \
    --config examples/qwen3vl_quant/configs/train.yaml

# Phase 2: PostQuant R4+GPTQ
python examples/qwen3vl_quant/ostquant_postquant.py
```

---

## 6. 数据校准

### 6.1 GPTQ 校准数据

使用文本数据即可 (纯 language model 的量化):

```python
# 选项 A: 通用文本数据
ds = load_dataset("hkust-nlp/deita-6k-v0", split="train[:256]")

# 选项 B: 自定义 VL 数据 (更贴合使用场景)
ds = load_dataset("lmms-lab/LLaVA-OneVision-Data", "FigureQA(MathV360K)", split="train[:128]")
```

### 6.2 OSTQuant 训练数据

同 GPTQ 校准数据，建议混合使用文本和 VL 数据:

```python
ds_text = load_dataset("hkust-nlp/deita-6k-v0", split="train[:128]")
ds_vl   = load_dataset("lmms-lab/LLaVA-OneVision-Data", ..., split="train[:128]")
ds = concatenate_datasets([ds_text, ds_vl])
```

---

## 7. 关键参数

| 参数 | QuaRot+GPTQ | OSTQuant+GPTQ |
|---|---|---|
| `transform_block_size_R1` | 1536 | 1536 |
| `transform_block_size_R4` | 256 | 256 |
| `transform_type` | `hadamard` | `random-hadamard` |
| `learnable` | False (默认) | True |
| `NUM_CALIBRATION_SAMPLES` | 256 | 256 |
| `MAX_SEQUENCE_LENGTH` | 2048 | 2048 |
| `sequential_targets` | `Qwen3VLTextDecoderLayer` | N/A (FSDP) |
| 训练步数 | N/A | 300 |
| 学习率 (旋转) | N/A | 0.136 |

---

## 8. 环境依赖

```bash
# 自定义 compressed-tensors (支持 SpinQuant + SmoothTransform)
pip install git+https://github.com/zhanglei1172/compressed-tensors.git@510d67d7573b10347aaf4a8d4dacef8cc4842ee9

# llm-compressor (本仓库)
pip install -e .

# transformers (支持 Qwen3VLForConditionalGeneration)
pip install transformers>=4.57.0

# 其他依赖
pip install qwen-vl-utils easydict loguru trl datasets
pip install flash-attn --no-build-isolation  # 可选，加速 attention
```

---

## 9. 文件清单

```
examples/qwen3vl_quant/
├── configs/
│   ├── quarot_gptq.yaml          # QuaRot(R1+R2+R4) + GPTQ W4
│   ├── ostquant_train.yaml       # OSTQuant Phase1: R1+R2 learnable + fake-quant
│   ├── r4_gptq.yaml              # OSTQuant Phase2: R4 + GPTQ W4
│   └── train.yaml                # OSTQuant FSDP 训练参数
├── quarot_gptq.py                # Step 1: QuaRot + GPTQ (单卡)
├── ostquant_train.py             # Step 2 Phase1: OSTQuant 训练 (多卡)
└── ostquant_postquant.py         # Step 2 Phase2: R4 + GPTQ 后量化 (单卡)
```

---

## 10. 注意事项

1. **q_norm / k_norm**: Qwen3-VL 新增的 attention QK norm (RMSNorm on head_dim)。它们不在量化范围内，SpinQuant 旋转也不影响它们，可安全忽略。
2. **Vision 模块**: 初始阶段只量化 language model 部分 (`ignore: ["re:.*visual.*"]`)。ViT 量化可后续单独进行。
3. **R1 block_size**: 必须设为 hidden_size (1536)，否则旋转维度不匹配。
4. **sequential_targets**: GPTQ 的 sequential onloading 需要指定 `Qwen3VLTextDecoderLayer` (不是 Qwen2.5-VL 的 `Qwen2_5_VLDecoderLayer`)。
5. **模型路径**: 需替换为实际的模型路径 (`MODEL_ID`) 和保存路径 (`SAVE_DIR`)。
