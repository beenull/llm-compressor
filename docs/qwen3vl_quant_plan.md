# Qwen3-VL 量化方案

> 基于 llm-compressor 框架对 Qwen3-VL 2B 进行 QuaRot+GPTQ 和 OSTQuant+GPTQ 量化的完整方案。

---

## 1. 目标

| 步骤 | 方法 | 说明 |
|---|---|---|
| Step 1 | **QuaRot + GPTQ** | 非学习旋转 (R1+R2+R4) + GPTQ W4 量化，作为 baseline |
| Step 2 | **OSTQuant + GPTQ** | 可学习旋转 (R1+R2 STE) + R4+GPTQ 后量化，更高精度 |

---

## 2. Qwen3-VL 2B Internal 模型结构

> 以下结构严格对应 `xllm-evaluation/docs/qwen3_vl_internal_infer_pipeline.md` 中的实际架构。

### 2.1 整体架构

实际使用的不是裸 `Qwen3VLForConditionalGeneration`，而是 **`OmniQwen3VLMedusaModel`** 包装器：

Qwen3-VL 2B Internal 本质上与开源 Qwen3-VL-2B-Instruct 区别不大，**语言模型部分完全复用**，主要差异在于：

- **骨干网络**: 复用 Qwen3-VL-2B-Instruct 的 language model (decoder-only transformer)，**ViT 视觉编码器未启用**
- **词表扩展**: 原始文本词表 + audio token (`<audio_0>` ~ `<audio_5120+>`)，统一词表编解码
- **MTP 加速**: 在骨干 LM 之上增加 Medusa 多头 (4 heads)，推测解码每步预测 5 个 token
- **不包含 audio tokenizer**: audio token 由外部预处理，作为文本形式输入

**量化影响**: 由于语言模型部分完全复用，量化流程与直接量化开源 Qwen3-VL-2B 无本质区别。Medusa heads 和扩展词表不参与量化。

### 2.2 三件套输入

模型加载需要三个独立组件：

| 组件 | 参数名 | 说明 |
|---|---|---|
| **HF 模型结构** | `--omni-model-name` | 标准 Qwen3-VL-2B-Instruct，提供模型结构+config，用 `from_pretrained` 加载后被 checkpoint 覆盖 |
| **扩展 Tokenizer** | `--omni-model-tokenizer` | 在原始 Qwen3-VL tokenizer 基础上扩展了 audio token 的词表 |
| **自定义 Checkpoint** | `--omni-model-ckpt` | 包含训练后的 language_model 权重 + lm_head (扩展词表) + medusa_head 权重 |

**加载顺序**:
1. `Qwen3VLForConditionalGeneration.from_pretrained(omni_model_name)` → 标准 HF 模型
2. 从 checkpoint 的 `lm_head.weight` 推断 `vocab_size` 和 `hidden_size`
3. `resize_token_embeddings(vocab_size)` → 扩展 embed_tokens 和 lm_head 到新词表大小
4. 构建 `OmniQwen3VLMedusaModel` 包装器 + Medusa heads
5. `load_state_dict(checkpoint, strict=False)` → 加载所有权重

### 2.3 完整模型层级路径

```
OmniQwen3VLMedusaModel (nn.Module)                    ← 我们 build_model() 返回的对象
│
├── language_model: Qwen3VLForConditionalGeneration    ← 量化目标 (omni_model.language_model)
│   ├── visual: Qwen3VLVisionModel                    ← ⚠️ 未使用，ckpt 中跳过
│   │   ├── patch_embed / pos_embed
│   │   ├── blocks[i]: Qwen3VLVisionBlock
│   │   │   ├── norm1 / norm2 (LayerNorm)
│   │   │   ├── attn.qkv / attn.proj
│   │   │   └── mlp.linear_fc1 / linear_fc2
│   │   ├── merger                                     ← SpinQuant mm_proj 指向这里
│   │   └── deepstack_merger_list                      ★ Qwen3-VL 新增
│   ├── model: Qwen3VLModel
│   │   └── language_model: Qwen3VLTextModel           ← 核心 decoder
│   │       ├── embed_tokens: Embedding(vocab_size_extended, 1536)
│   │       ├── layers[i]: Qwen3VLTextDecoderLayer × 28
│   │       │   ├── input_layernorm (RMSNorm)
│   │       │   ├── self_attn
│   │       │   │   ├── q_proj / k_proj / v_proj / o_proj
│   │       │   │   └── q_norm / k_norm                ★ Qwen3-VL 新增 (RMSNorm on head_dim)
│   │       │   ├── post_attention_layernorm (RMSNorm)
│   │       │   └── mlp
│   │       │       └── gate_proj / up_proj / down_proj
│   │       ├── norm (RMSNorm)
│   │       └── rotary_emb
│   └── lm_head: Linear(1536, vocab_size_extended, bias=False)
│
└── medusa_head: ModuleList[4 个推测解码头]             ← 不参与量化
    └── Sequential:
        ├── ResBlock(1536)     ← 残差块: x + SiLU(Linear(x)), 零初始化
        └── Linear(1536, vocab_size_extended, bias=False)
```

### 2.4 关键维度

| 参数 | 值 |
|---|---|
| 外层包装类名 | `OmniQwen3VLMedusaModel` |
| 骨干模型类名 | `Qwen3VLForConditionalGeneration` |
| hidden_size | 1536 |
| num_layers | 28 |
| num_attention_heads | 12 |
| num_key_value_heads | 2 (GQA) |
| intermediate_size | 8960 |
| vocab_size (原始) | 151665 |
| vocab_size (扩展后) | ~156785+ (从 ckpt lm_head 推断) |
| Decoder Layer 类名 | `Qwen3VLTextDecoderLayer` |
| Norm 类型 | RMSNorm |
| medusa_heads | 4 |
| medusa_layers (ResBlock) | 1 |

### 2.5 Checkpoint Key 映射

Checkpoint key 格式与 HF 模型不完全一致，`_remap_checkpoint_keys` 执行以下变换：

```python
# 1. 跳过视觉编码器
if key.startswith("language_model.visual.") or key.startswith("vision_encoder"):
    continue  # 不加载 ViT 权重

# 2. Language model 权重重映射 (多了一层 language_model)
"language_model.model.xxx" → "language_model.model.language_model.xxx"
# 例: ckpt 中 language_model.model.layers.0.self_attn.q_proj.weight
# → HF 中 language_model.model.language_model.layers.0.self_attn.q_proj.weight

# 3. 其他 key 保持不变
"language_model.lm_head.weight" → 不变
"medusa_head.0.0.linear.weight" → 不变
```

### 2.6 量化目标与量化无关部分

| 组件 | 路径 | 量化处理 |
|---|---|---|
| **Language Decoder** | `language_model.model.language_model.layers[i]` | ✅ 量化目标 (SpinQuant + GPTQ) |
| **lm_head** | `language_model.lm_head` | ❌ ignore (SpinQuant 映射中引用但不量化) |
| **embed_tokens** | `language_model.model.language_model.embed_tokens` | ❌ ignore |
| **ViT Visual** | `language_model.visual.*` | ❌ ignore (`re:.*visual.*`) |
| **Medusa Heads** | `medusa_head.*` | ❌ 不参与量化 (在 OmniModel 层，不在 HF model 内) |
| **q_norm / k_norm** | `language_model.model.language_model.layers[i].self_attn.{q,k}_norm` | ❌ 安全忽略 (RMSNorm on head_dim，SpinQuant 旋转不影响) |

**关键**: `quarot_gptq.py` 中提取 `hf_model = omni_model.language_model` 后，对 `hf_model` (即 `Qwen3VLForConditionalGeneration`) 做量化。Medusa heads 在外层 `OmniQwen3VLMedusaModel` 上，不受影响。

### 2.7 与 Qwen2.5-VL 的关键差异

| 特性 | Qwen2.5-VL | Qwen3-VL (内部 Omni) |
|---|---|---|
| 实际使用模型 | 裸 HF model | `OmniQwen3VLMedusaModel` 包装 |
| 量化骨干路径 | `model.language_model.layers.{i}` | **完全一致** ✅ |
| Decoder Layer 类名 | `Qwen2_5_VLDecoderLayer` | `Qwen3VLDecoderLayer` |
| QK Norm | ❌ | ✅ `q_norm` + `k_norm` |
| Vision DeepStack | ❌ | ✅ `deepstack_merger_list` |
| 词表 | 原始 | 扩展 (audio tokens) |
| Medusa MTP | ❌ | ✅ 4 heads (不参与量化) |
| SpinQuant regex | `re:.*language_model.*q_proj$` | **可复用** ✅ |

**结论**: SpinQuant/Norm 映射 regex 可直接从 Qwen2.5-VL 复用，只需更改注册 key。量化时需注意通过 `omni_model.language_model` 提取 HF 骨干模型。

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

### 3.3 映射说明：旋转 vs 量化的区别

**关键概念**: SpinQuant 映射中注册的 `embedding` 和 `lm_head` 是用于**旋转**（R1 scheme），不是用于量化。

| 映射字段 | 指向模块 | 旋转 (SpinQuant) | 量化 (GPTQ) |
|---|---|---|---|
| `embedding` | `embed_tokens` | ✅ R1 旋转：`_center_embeddings()` 均值归零后乘 Hadamard 矩阵，让后续层输入分布更均匀 | ❌ 不量化（Embedding 层，不是 Linear） |
| `lm_head` | `lm_head` | ✅ R1 旋转：作为 R1 scheme 尾端，保证旋转对称性 | ❌ 不量化（在 GPTQ ignore 列表中） |
| `attn_q/k/v/o` | `q/k/v/o_proj` | ✅ R1+R2 旋转 | ✅ GPTQ W4 量化 |
| `mlp_in/out` | `up/gate/down_proj` | ✅ R1+R4 旋转 | ✅ GPTQ W4 量化 |
| `mm_proj` | `visual.merger.linear_fc2` | ✅ R1 旋转（视觉投影尾端） | ❌ 不量化（在 `re:.*visual.*` ignore 中） |

### 3.4 QuaRot 与 SpinQuant 的关系

框架中**没有独立的 QuaRotModifier**，统一用 `SpinQuantModifier`，通过 `learnable` 参数切换模式：

| 参数 | QuaRot（非学习） | OSTQuant / SpinQuant（可学习） |
|---|---|---|
| `learnable` | `False`（默认值，YAML 中不写即为 False） | `True` |
| `transform_type` | `"hadamard"`（固定 Hadamard 矩阵） | `"random-hadamard"`（随机初始化，可学习） |
| 是否需要训练 | ❌ datafree（旋转步骤不需要数据） | ✅ FSDP + KL 蒸馏训练 |
| 旋转矩阵 | 固定，不更新 | 通过 STE 梯度在 Stiefel 流形上优化 |
| 对应配置 | `quarot_gptq.yaml` | `ostquant_train.yaml` |

因此 `quarot_gptq.py` 和 `ostquant_train.py` 都需要 `from llmcompressor.modifiers.transform.spinquant import mappings, norm_mappings` 来注册模型映射，这是标准做法。v5/Qwen2.5-VL 的官方例子也是同样模式。

### 3.5 映射注册的调用机制

`quarot_gptq.py` 中看不到显式调用 mappings 的代码，但注册到全局 dict 后会被 `oneshot()` 内部自动使用。完整调用链路：

```
① quarot_gptq.py 顶层执行:
   mappings.SPINQUANT_MAPPING_REGISTRY["Qwen3VLForConditionalGeneration"] = SpinQuantMapping(...)
   norm_mappings.NORM_MAPPING_REGISTRY["Qwen3VLForConditionalGeneration"] = [...]
       ↓ (写入全局 dict，Python 模块级别的可变对象)

② oneshot(model=hf_model, recipe="quarot_gptq.yaml") 内部:
       ↓
   SpinQuantModifier.on_initialize(state)     # state.model = hf_model
       ↓
   self.mappings = infer_mapping_from_model(state.model)
       ↓
   infer_mapping_from_model() 内部:
       architecture = model.__class__.__name__   # → "Qwen3VLForConditionalGeneration"
       return SPINQUANT_MAPPING_REGISTRY.get(architecture, _default_mappings)
       # → 命中步骤①注册的映射
       ↓
   self.norm_mappings = infer_norm_mapping_from_model(state.model)  # 同理
       ↓
③ 用 self.mappings 构建各旋转 scheme:
   _create_r1_scheme() → 引用 mappings.embedding, attn_o, mlp_out, attn_q/k/v, mlp_in, lm_head
   _create_r2_scheme() → 引用 mappings.attn_v, attn_o
   _create_r4_scheme() → 引用 mappings.mlp_in, mlp_out
       ↓
④ on_start() 中:
   _center_embeddings() → 使用 mappings.embedding 定位 embed_tokens 做均值归零
   _fuse_norms()         → 使用 norm_mappings 定位 RMSNorm 融入相邻 Linear
   apply_transform_config() → 插入旋转矩阵到对应层
```

**关键**: 映射注册必须在 `oneshot()` 调用之前完成，因为 `on_initialize` 在 `oneshot` 内部第一步就会查询 registry。

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

使用纯文本数据即可，对 Internal 模型没有负面影响。原因：

1. **量化目标是 `hf_model`**（`Qwen3VLForConditionalGeneration`），不是 `OmniQwen3VLMedusaModel`。GPTQ 校准只需 `hf_model.forward(input_ids)` 跑通即可
2. **语言模型完全复用**。Internal 模型的 language decoder 与开源 Qwen3-VL-2B 一模一样，文本 token 走的路径完全相同
3. **audio token 只是扩展词表中的普通 token**。`<audio_667>` 编码后就是一个 vocab id，走 `embed_tokens → decoder layers → lm_head` 的路径与文本 token 完全一致
4. **v5 例子也用文本数据**。`v5_example.py` 对 Qwen2.5-VL 的 GPTQ 量化同样用纯文本校准

```python
# 当前方案: 通用文本数据 (已验证可行)
ds = load_dataset("hkust-nlp/deita-6k-v0", split="train[:256]")
```

### 6.2 可选优化: 混合 audio 数据校准

如需让量化对 audio 场景更精准，可混入 audio token 数据做校准（**TODO: 跑通全流程后评估是否需要**）：

```python
# 可选: 构造包含 audio token 的校准样本
audio_prompts = ["<|ATQTA|>...<audio_667><audio_993>..."]
ds_audio = tokenize_audio_prompts(audio_prompts)
ds = concatenate_datasets([ds_text, ds_audio])
```

### 6.3 OSTQuant 训练数据

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

---

## 11. 云端部署调试记录 (XP A100)

### 11.1 云端环境信息

| 项目 | 值 |
|---|---|
| 平台 | XP A100 (cnwlb-a100-p01046) |
| Python | 3.10 |
| llm-compressor | `/workspace/gaoy25@xiaopeng.com/gytmp/quant/llm-compressor` |
| compressed-tensors | `/workspace/gaoy25@xiaopeng.com/gytmp/quant/compressed-tensors` (自定义 fork) |

### 11.2 云端模型路径

```
--omni-model-name      /workspace/gaoy25@xiaopeng.com/model/qwen3_vl/Qwen3-VL-2B-Instruct
--omni-model-tokenizer /workspace/gaoy25@xiaopeng.com/model/group_share/adc-perception-mlinfra/shijh2/qwen3_vl_extend
--omni-model-ckpt      /workspace/gaoy25@xiaopeng.com/model/group_share/adc-perception-mlinfra/malf/omni/hf2aif_0304_final_resave.pt
```

### 11.3 环境安装步骤

```bash
# 1. compressed-tensors (自定义 SpinQuant fork)
cd /workspace/gaoy25@xiaopeng.com/gytmp/quant/compressed-tensors
pip install -e .

# 2. llm-compressor
cd /workspace/gaoy25@xiaopeng.com/gytmp/quant/llm-compressor
pip install -e .

# 3. transformers >= 4.57.0 (支持 Qwen3VLForConditionalGeneration)
pip install transformers>=4.57.0

# 4. numpy 版本锁定 (解决 scipy/pandas 兼容性)
pip install numpy==1.26.4 scipy pandas --force-reinstall

# 5. 其他依赖
pip install datasets qwen-vl-utils easydict loguru trl
```

### 11.4 遇到的问题及解决

| # | 错误 | 原因 | 解决 |
|---|---|---|---|
| 1 | `ModuleNotFoundError: No module named 'datasets'` | 未安装 | `pip install datasets` |
| 2 | `ModuleNotFoundError: No module named 'llmcompressor'` | 未安装 | `cd llm-compressor && pip install -e .` |
| 3 | `numpy.dtype size changed, expected 96 from C header, got 88` | numpy 版本与编译的 scipy/pandas 不兼容 | `pip install numpy==1.26.4 scipy pandas --force-reinstall` |
| 4 | `ImportError: attempted relative import with no known parent package` | `build_model.py` 使用 `from .omni_qwen3vl_medusa import` 但通过 `sys.path` 调用 | 改为 `from omni_qwen3vl_medusa import` (绝对导入) |
| 5 | `HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name'` | `AutoTokenizer.from_pretrained()` 把本地路径当 HF Hub repo_id 校验 | 加 `local_files_only=True` |

### 11.5 DEBUG 模式

脚本内置 DEBUG 开关，通过环境变量控制：

```bash
# 静默模式 (默认) — 无调试输出
python examples/qwen3vl_quant/quarot_gptq.py --skip-quant

# 调试模式 — 输出所有 step 日志、prompt 内容、生成结果
DEBUG=true python examples/qwen3vl_quant/quarot_gptq.py --skip-quant
```

实现方式：
- `quarot_gptq.py`: 所有 `print()` 替换为 `dprint()`，仅 `DEBUG=true` 时输出；非 DEBUG 时 `logging.disable(logging.INFO)` 屏蔽库日志
- `build_prompt.py`: 非 DEBUG 时移除 loguru 默认 handler，设为 WARNING 级别，屏蔽 `logger.info()` 的 prompt 打印

### 11.6 验证通过的命令

```bash
# --skip-quant 模式: 构建内部 Omni 模型 → ATQTA 推理测试 (不做量化)
DEBUG=true python examples/qwen3vl_quant/quarot_gptq.py --skip-quant
```

### 11.7 文件清单 (更新)

```
examples/qwen3vl_quant/
├── __init__.py                   # 空 init
├── configs/
│   ├── quarot_gptq.yaml          # QuaRot(R1+R2+R4) + GPTQ W4
│   ├── ostquant_train.yaml       # OSTQuant Phase1: R1+R2 learnable + fake-quant
│   ├── r4_gptq.yaml              # OSTQuant Phase2: R4 + GPTQ W4
│   └── train.yaml                # OSTQuant FSDP 训练参数
├── omni_qwen3vl_medusa.py        # OmniQwen3VLMedusaModel 类 (从 xllm-evaluation 拷贝)
├── build_model.py                # 模型构建 (从 xllm-evaluation 拷贝, 已改为绝对导入)
├── build_prompt.py               # Prompt 构建 (从 xllm-evaluation 拷贝, 含 DEBUG 控制)
├── quarot_gptq.py                # Step 1: QuaRot + GPTQ (单卡, 含 DEBUG 模式)
├── ostquant_train.py             # Step 2 Phase1: OSTQuant 训练 (多卡)
└── ostquant_postquant.py         # Step 2 Phase2: R4 + GPTQ 后量化 (单卡)
```

---

## 12. TODO List

- [ ] **跑通 QuaRot+GPTQ 全流程** — 去掉 `--skip-quant`，运行完整量化 + 保存
- [ ] **量化后 ATQTA 推理验证** — 确认量化后模型生成质量
- [ ] **评估是否需要混合 audio 数据做 GPTQ 校准** — 对比纯文本 vs 混合数据的量化精度
- [ ] **ostquant_train.py 适配内部 Omni 模型** — 改用 `build_model()` 加载，提取 `.language_model`
- [ ] **ostquant_postquant.py 适配** — tokenizer 路径等更新
- [ ] **OSTQuant Phase 1 训练** — FSDP 多卡训练 R1+R2
- [ ] **OSTQuant Phase 2 后量化** — R4+GPTQ
- [ ] **最终精度评估** — QuaRot vs OSTQuant 量化后精度对比
