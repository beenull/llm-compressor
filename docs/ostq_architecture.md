# OSTQuant 完整架构分析 (基于 talker_ostquant.py)

> **参考实现**: `examples/omnitalker/talker_ostquant.py` — 完整 OSTQuant (含 SmoothTransform S1-S4)
> **简化实现**: `examples/v5/ostq.py` — 仅含 R1+R2 旋转，**无 smooth factor**（详见末尾对比）

---

## 1. OSTQuant 核心公式

**OSTQuant = SpinQuant (可学习正交旋转矩阵 R1/R2) + SmoothTransform (可学习对角缩放 S1-S4) + 伪量化 (STE) + 知识蒸馏 (KL)**

模型权重 W 全程冻结 (`requires_grad=False`)，仅训练:
- **R1/R2** (2D 正交矩阵): Stiefel 流形上优化，保持正交性
- **S1/S2/S3/S4** (1D 对角缩放): 普通 SGD 优化，初始化为全 1

数学表达 (以线性层 $Y=XW$ 为例):
$$Y = X \cdot S^{-1} \cdot R \cdot W \cdot R^T \cdot S$$

其中 $R$ 是正交旋转矩阵, $S$ 是对角缩放矩阵. 训练目标:
$$\min_{R, S} \text{KL}\big( f_{teacher}(x),\ f_{quant}(x; R, S) \big)$$

---

## 2. 完整 Pipeline

```
原始 Qwen3OmniMoeTalker HF 权重
    │
    ▼  [Step1] talker_ostquant.py — OSTQuant 训练 (多卡 FSDP)
    │
    │   ❶ pre_trans_talker():
    │      SpinQuant R1+R2 可学习旋转
    │      + SmoothTransform S1/S2/S3/S4 可学习缩放
    │      + fuse_norms + untie_embeddings
    │
    │   ❷ pre_compression_talker():
    │      QuantizationModifier(W4A8, STE=True) 伪量化
    │
    │   ❸ fsdp_main():
    │      FSDP 多卡训练 (MyTrainer)
    │      知识蒸馏: KL(student, teacher)
    │      只更新 R1/R2 + S1-S4, 模型权重冻结
    │
    │   ❹ fold_transforms_into_weights():
    │      R_learned · W → W_folded (旋转永久融入权重)
    │
    │   ❺ post_compression_talker():
    │      保存模型 + transform_state_dict.pt + norm_weight.pt
    │
    ▼  输出: FP 权重 (已融入 R1/R2+S1-S4) + transform 状态
    │
    ▼  [Step2] v5_example.py — Postquant 校准 (单卡)
    │   SpinQuant R4 + W4A8 MSE observer 校准
    │
    ▼  输出: 最终量化模型 (realq/fq/trans)
```

---

## 3. SmoothTransform 类详解

```python
class SmoothTransform(TransformBase):
    def __init__(self, dim, inverse=False, is_out=True, is_qk=False, head_dim=-1):
        self.scale = nn.Parameter(torch.ones(dim))   # 1D 可学习缩放, 初始化为 1
        self.inverse = inverse                        # 是否取倒数
        self.is_out = is_out                          # 作用在 output 维度 (weight[:, :]) vs input 维度
        self.is_qk = is_qk                            # S3 专用: 对 Q/K proj 做 repeat
        self.head_dim = head_dim                       # S2 专用: 按 head 维度 repeat
```

**核心原理**: 对称缩放对 —— 在相邻的两个模块间插入 $S$ 和 $S^{-1}$，使两者各自的量化误差最小化:

```
前一层 ──→ × S^{-1} ──→ [量化] ──→ × S ──→ 后一层
```

`forward` 逻辑:
- `is_out=True`: `scale` reshape 为 `(-1, 1)` 乘在 weight 的 out_features 维度上
- `is_out=False, inverse=True`: 用 `1/scale` 乘在 weight 的 in_features 维度上
- `is_qk=True`: `scale(dim)` → `repeat([1,2])` → `(2*dim,)`, 适配 Q/K head 结构
- `head_dim != -1`: `repeat_interleave` 按 head_dim 展开, 适配 V/O 的 GQA 结构

通过 `torch.nn.utils.parametrize.register_parametrization(module, "weight", transform)` 注册:
- forward 时: `module.weight` 自动变为 `transform(module.parametrizations.weight.original)`
- `right_inverse()`: 提供初始化时的逆变换

---

## 4. S1-S4 缩放因子映射

### 论文概念 → 代码映射

| 缩放因子 | 论文含义 | 作用位置 | 代码实现 | 维度 |
|---|---|---|---|---|
| **S1** (attn) | LayerNorm → QKV 间的缩放 | `input_layernorm` ←→ `q/k/v_proj` | `S1^{-1}` on `input_layernorm`, `S1` on `q/k/v_proj` | `hidden_size` |
| **S1** (mlp) | LayerNorm → FFN 间的缩放 | `post_attention_layernorm` ←→ `gate/up_proj` | **已注释掉** (代码中 mlp S1 被注释) | `hidden_size` |
| **S2** | V/O projection 间的缩放 | `v_proj` ←→ `o_proj` | `S2` on `v_proj(out)`, `S2^{-1}` on `o_proj(in)` | `v_proj.out_features` (带 head_dim) |
| **S3** | Q/K projection 的缩放 | `q_proj` ←→ `k_proj` | **已注释掉** (代码中 S3 register 被注释) | `k_proj.out_features // 2` (is_qk) |
| **S4** | FFN up/down 间的缩放 | `up_proj` ←→ `down_proj` | `S4` on `up_proj(out)`, `S4^{-1}` on `down_proj(in)` | `up_proj.out_features` |

### 实际生效的缩放 (`ENABLE_SMOOTH=True`):
- ✅ **S1 (attn)**: `input_layernorm ←→ q/k/v_proj`
- ❌ **S1 (mlp)**: `post_attention_layernorm ←→ gate/up_proj` (已注释)
- ✅ **S2**: `v_proj ←→ o_proj` (带 head_dim repeat)
- ❌ **S3**: `q_proj ←→ k_proj` (已注释)
- ✅ **S4**: `up_proj ←→ down_proj` (对每个 expert 独立)

### S2 的 GQA head_dim 处理

V_proj 和 O_proj 在 GQA 中维度不同 (num_kv_heads != num_heads):

```python
# S2 维度 = v_proj.out_features, 按 head_dim 分组
S2_transform = SmoothTransform(v_proj.out_features, is_out=True, head_dim=head_dim)
# S2_inv 维度相同, 但 is_out=False, 在 O_proj 的 in_features 维度上用 repeat_interleave 展开
S2_transform_inv = SmoothTransform(v_proj.out_features, is_out=False, inverse=True, head_dim=head_dim)
S2_transform_inv.scale = S2_transform.scale  # 共享同一个 scale 参数!
```

---

## 5. pre_trans_talker() 详细流程

```python
def pre_trans_talker(model):
    # ❶ SpinQuant R1+R2 旋转
    state = State(model=model.talker)
    recipe_ = [SpinQuantModifier(
        backe_mean=False,
        learnable=True,
        rotations=["R1", "R2"],
        transform_block_size_R1=1024,
        transform_type="random-hadamard",
    )]
    # on_initialize → 创建旋转方案
    # untie_word_embeddings → 避免 embedding 被意外修改
    # _fuse_norms → RMSNorm 权重融入相邻 Linear

    # ❷ 如果 ENABLE_SMOOTH: 注册 S2/S4 缩放 (第一轮)
    for each (q, k, v, o, up, down) layer:
        # S2: v_proj ←→ o_proj (带 head_dim)
        S2 = SmoothTransform(v_out, is_out=True, head_dim=hd)
        S2_inv = SmoothTransform(v_out, is_out=False, inverse=True, head_dim=hd)
        S2_inv.scale = S2.scale  # 共享参数
        register_parametrization(v_proj, "weight", S2)
        register_parametrization(o_proj, "weight", S2_inv)

        # S4: up_proj ←→ down_proj (每个 expert 独立)
        for each (up, down) expert:
            S4 = SmoothTransform(up_out, is_out=True)
            S4_inv = SmoothTransform(down_in, is_out=False, inverse=True)
            S4_inv.scale = S4.scale  # 共享参数
            register_parametrization(up_proj, "weight", S4)
            register_parametrization(down_proj, "weight", S4_inv)

    # ❸ apply_transform_config → 实际插入 R1/R2 旋转矩阵

    # ❹ 如果 ENABLE_SMOOTH: 注册 S1 缩放 (第二轮, 在 R1/R2 之后)
    for each (attn_norm, q, k, v, mlp_norm, gate, up) layer:
        # S1 (attn): input_layernorm ←→ q/k/v_proj
        S1 = SmoothTransform(hidden_size, is_out=False)
        S1_inv = SmoothTransform(hidden_size, is_out=True, inverse=True)
        S1_inv.scale = S1.scale  # 共享参数
        register_parametrization(attn_norm, "weight", S1_inv)
        register_parametrization(q_proj, "weight", S1)
        register_parametrization(k_proj, "weight", S1)
        register_parametrization(v_proj, "weight", S1)
        # ⚠️ S1 (mlp): post_attention_layernorm ←→ gate/up_proj → 已注释

    return state, recipe_, model
```

**注意注册顺序**: S2/S4 先于 `apply_transform_config(R1/R2)`, S1 后于 R1/R2。
这影响 `parametrizations` 的堆叠顺序:
- v_proj: `original → S2 → R1 → S1` (parametrize 按注册顺序链式调用)
- o_proj: `original → S2_inv → R1`

---

## 6. pre_compression_talker() — 伪量化配置

```python
def pre_compression_talker(model):
    # 对 model.talker 注册 QuantizationModifier
    QuantizationModifier(
        config_groups={
            "group_0": {  # talker 主模型 - W4A8
                "weights": INT4 symmetric channel,
                "input_activations": INT8 symmetric tensor,
                "targets": [q/k/v/o/up/gate/out_proj],
                "ste": True,
            },
            "group_1": {  # talker down_proj - W4A16 (激活不量化)
                "weights": INT4,
                "input_activations": INT16,  # 16bit ≈ 不量化
                "targets": [down_proj],
                "ste": True,
            },
            "group_2": {  # code_predictor - W8A8
                "weights": INT8,
                "input_activations": INT8,
                "targets": [code_predictor q/k/v/o/out_proj],
                "ste": True,
            },
            "group_3": {  # code_predictor down_proj - W8A16
                "weights": INT8,
                "input_activations": INT16,
                "targets": [code_predictor down_proj],
                "ste": True,
            },
        }
    )
    model.talker.apply(enable_quantization)  # 启用伪量化
```

---

## 7. fsdp_main() — 分布式训练

### 训练流程

```
                   ┌── Teacher Model (frozen, FP) ──┐
                   │                                 │
Input Data ──→ Student Model (R+S learnable) ──→ KL Loss
               │ fake-quant W4A8 (STE)      │
               │ model weights frozen        │
               └─ forward-backward ──────────┘
                    ↓
            只更新 R1/R2 (Stiefel) + S1/S2/S4 (SGD)
```

### 关键组件

| 组件 | 说明 |
|---|---|
| **TalkerTrainer** | 继承 `MyTrainer`, 自定义 `compute_loss` 处理 TTS 数据 |
| **Teacher 模型** | 原始模型 deepcopy, `thinker.model.layers` 已删除 (只保留 talker), frozen |
| **损失函数** | `kl_top`: 取 teacher logits 的 top-k, 与 student 对应位置做 KL 散度 |
| **ignored_modules** | thinker embeddings, talker embeddings, codec_head, code_predictor, text_projection, model.norm — 均不参与 FSDP 分片 |

### 训练参数分组 (MyTrainer.create_optimizer_and_scheduler)

```python
for param in model.parameters():
    if param.requires_grad:
        if len(param.size()) == 1:       # 1D → S1/S2/S4 smooth factors
            params_smooth.append(param)
        else:                            # 2D → R1/R2 旋转矩阵
            params_rotate.append(param)
```

| 参数组 | 内容 | 优化器 | 学习率 | 约束 |
|---|---|---|---|---|
| `params_rotate` | R1/R2 旋转矩阵 | SGDG / Riemannian SGD/Adam | `rotate_lr` (~0.136) | **Stiefel 流形** (geoopt ManifoldParameter) |
| `params_smooth` | S1/S2/S4 缩放因子 | 普通 SGD | `smooth_lr` (~0.0098) | 无约束, momentum=0.9 |

### compute_loss 流程 (TTS 特化)

1. `model.thinker` forward (frozen, `torch.no_grad`) → 获取 hidden_states
2. 逐样本构建 talker 输入 (prefix + codec embeddings, teacher forcing)
3. `model.talker.model` forward → 获取 logits
4. 同时用 teacher 模型获取 teacher logits
5. `kl_top` loss: top-k KL 散度, masking `labels==-100`

---

## 8. 训练后保存流程

### main 执行序列

```python
if __name__ == "__main__":
    setup()  # dist.init_process_group("nccl")
    model, processor = dist_load_model(load_processor=True)

    # 保存原始 norm weight (训练前, 供后续 postquant 使用)
    save_norm_weight = model.talker.model.norm.weight.float().cpu()

    with patch_module_non_persistent_buffers(model):
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        state_text, recipe_trans, model = pre_trans_talker(model)    # ❶ R1/R2 + S1-S4
        state, recipe_, model = pre_compression_talker(model)         # ❷ fake-quant
        fsdp_main(model, config)                                       # ❸ 训练

    if rank == 0:
        recipe_trans[0]._fold_transforms_into_weights(state_text.model)  # ❹ fold R into W
        post_compression_talker(state, recipe_, model, processor,        # ❺ 保存
                                additional_tensors={"save_norm_weight": ...})
    cleanup()
```

### post_compression_talker 保存内容

| 输出文件 | 内容 |
|---|---|
| `SAVE_DIR/` (model.save_pretrained) | FP 权重, R1/R2 已 fold 进去, smooth factor 已参数化在 weight 里 |
| `SAVE_DIR/transform_state_dict.pt` | 所有 TransformBase 模块的 state_dict (R1/R2/S1-S4 的 scale 参数) |
| `SAVE_DIR/norm_weight.pt` | 训练前的原始 RMSNorm 权重 (fuse_norms 前保存的) |

---

## 9. SpinQuant R1-R4 旋转 → 论文映射

| 旋转 | 论文名 | 类型 | 位置 | 用途 |
|---|---|---|---|---|
| **R1** | $R_{res}$ | 离线 (offline) | 全连接层间 (weight_output/weight_input) | 减少跨层量化误差传播 |
| **R2** | $R_{ov}^h$ | 离线 (offline) | 注意力头内 (block_wise V↔O) | 减少头内量化误差 |
| **R3** | $R_{qk}$ | 在线 (online) | Q/K KV cache | 运行时应用 |
| **R4** | $R_{down}$ | 在线 (online) | FFN down_proj 激活 | 运行时应用, block_size=256 |

**talker_ostquant.py 中**: 训练 R1+R2 (离线), R3/R4 在 postquant 阶段处理

### SpinQuant 模型映射注册

```python
mappings.SPINQUANT_MAPPING_REGISTRY["Qwen3OmniMoeTalkerForConditionalGeneration"] = SpinQuantMapping(
    mm_proj=[r"re:.*text_projection\.linear_fc2$", r"re:.*hidden_projection\.linear_fc2$"],
    embedding=r"re:(model\.codec_embedding|.*codec_embedding\.\d+)$",
    attn="re:.*self_attn$",
    attn_q/k/v/o="re:.*model.*{q,k,v,o}_proj$",
    mlp_in=[r"re:.*up_proj$", r"re:.*mlp.*(gate|gate_proj)$"],
    mlp_out=[r"re:.*mlp.*down_proj$"],
    lm_head=r"re:.*(lm_head\.\d+|codec_head)$",
)
```

---

## 10. 训练时模型内部数据流

```
                S1^{-1}          R1          S1         fake-quant
input_layernorm ──→ ×scale^{-1} ──→ Rotate ──→ ×scale ──→ Q(W4A8) ──→ q/k/v_proj
                                                                        │
                          S2                R2              S2^{-1}     │
               v_proj ──→ ×scale ──→ HeadRotate ──→ ×scale^{-1} ──→ o_proj
                                                                        │
                S4                                   S4^{-1}           │
        up_proj ──→ ×scale ──→ [activation] ──→ ×scale^{-1} ──→ down_proj

所有 ×scale 操作通过 register_parametrization 自动执行
R1/R2 旋转通过 apply_transform_config 插入
fake-quant 通过 QuantizationModifier + enable_quantization 启用
```

**梯度流**: Loss → STE 穿过伪量化 → 到达 S 和 R 的 `nn.Parameter` → 更新

---

## 11. v5/ostq.py (简化版) vs talker_ostquant.py (完整版) 对比

| 特性 | `v5/ostq.py` (Qwen2.5-VL) | `talker_ostquant.py` (Qwen3OmniMoeTalker) |
|---|---|---|
| **目标模型** | `Qwen2_5_VLForConditionalGeneration` | `Qwen3OmniMoeForConditionalGeneration` 的 talker 子模型 |
| **旋转** | R1+R2 ✅ | R1+R2 ✅ |
| **SmoothTransform S1** | ❌ 不存在 | ✅ attn norm → QKV (mlp 已注释) |
| **SmoothTransform S2** | ❌ 不存在 | ✅ V ↔ O (带 head_dim) |
| **SmoothTransform S3** | ❌ 不存在 | ❌ 已注释 |
| **SmoothTransform S4** | ❌ 不存在 | ✅ up ↔ down (per expert) |
| **量化分组** | 2 组 (W4A8, W4A16) | 4 组 (talker W4A8/W4A16 + predictor W8A8/W8A16) |
| **数据类型** | 文本 + 图像 (VL) | TTS 音频 (codec codes) |
| **params_smooth** | 实际为**空** (RMSNorm 不触发 replace_ln) | 实际有值 (S1/S2/S4 的 scale 参数) |
| **R1 block_size** | 3584 (= Qwen2.5-VL hidden_size) | 1024 |
| **Teacher 结构** | 完整模型 deepcopy | 删除 `thinker.model.layers`, 只保留 talker |
| **postquant Step2** | `v5_example.py` (R4 + MSE) | 未在同文件实现 |

**关键发现**: v5/ostq.py 虽然配置了 `smooth_lr`，但因为 Qwen2.5-VL 使用 RMSNorm (不是 LayerNorm)，`replace_ln_to_rmsnorm` 不会触发，所以 `params_smooth` 实际为空列表 —— **不训练任何 smooth factor**。

---

## 12. 适配新模型 (如 Qwen3-VL) 需要修改的部分

| 需修改项 | 说明 |
|---|---|
| **模型类名** | `Qwen3VLForConditionalGeneration` |
| **SpinQuant 映射注册** | 新增 `SPINQUANT_MAPPING_REGISTRY["Qwen3VLForConditionalGeneration"]` |
| **Norm 映射注册** | 新增 `NORM_MAPPING_REGISTRY["Qwen3VLForConditionalGeneration"]` |
| **R1 block_size** | 设为 Qwen3-VL 的 hidden_size |
| **FSDP wrap 层** | `Qwen3VLTextDecoderLayer,TeacherModel` |
| **模型层路径前缀** | Qwen3-VL 可能无 `language_model` 前缀, 需验证 |
| **数据 pipeline** | VL 数据: 文本+图像, 非 TTS 音频 |
| **量化分组** | 调整 targets regex 匹配新模型的层名 |
| **compute_loss** | 改为标准 LM loss 或 VL loss, 非 TTS codec loss |

---

## 13. 运行命令

### talker_ostquant.py (完整版)
```bash
torchrun --nproc_per_node=2 examples/omnitalker/talker_ostquant.py --config <config.yaml>
```

### v5/ostq.py (简化版)
```bash
torchrun --nproc_per_node=2 examples/v5/ostq.py --config examples/v5/train.yaml
```

### Postquant (Step2)
```bash
python examples/v5/v5_example.py
```

---

## 14. 环境依赖

```bash
# 自定义 compressed-tensors (支持 spinquant/ostquant + SmoothTransform)
pip install git+https://github.com/zhanglei1172/compressed-tensors.git@510d67d7573b10347aaf4a8d4dacef8cc4842ee9

# llm-compressor
cd llm-compressor && pip install -e .

# 其他依赖
pip install qwen-omni-utils easydict loguru trl geoopt librosa soundfile
```
