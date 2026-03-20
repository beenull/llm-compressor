# SpinQuant / QuaRot 架构分析 (基于 llm-compressor)

> **源码**: `src/llmcompressor/modifiers/transform/spinquant/`
> **论文**: [SpinQuant: LLM quantization with learned rotations](https://arxiv.org/abs/2405.16406)
> **官方例子**: `examples/transform/spinquant_example.py`

---

## 1. 核心思想

SpinQuant 通过在模型的关键位置插入**正交旋转矩阵**，将权重和激活"旋转"到一个动态范围更小的空间中，从而降低量化损失。

数学表达：对于线性层 $Y = XW$，插入旋转后变为：
$$Y = (X \cdot R) \cdot (R^T \cdot W)$$

由于 $R$ 是正交矩阵 ($R \cdot R^T = I$)，数学上等价，但旋转后的 $R^T \cdot W$ 分布更均匀，量化误差更小。

### QuaRot vs SpinQuant vs OSTQuant

| 方法 | 旋转矩阵类型 | 是否可学习 | 配置 | 说明 |
|---|---|---|---|---|
| **QuaRot** | 固定 Hadamard | ❌ `learnable=False` | `transform_type="hadamard"` | 非学习，datafree |
| **SpinQuant** | 可学习正交矩阵 | ✅ `learnable=True` | `transform_type="random-hadamard"` | Stiefel 流形优化 |
| **OSTQuant** | SpinQuant + SmoothTransform | ✅ | SpinQuant R1/R2 + S1-S4 | 额外加对角缩放 |

**框架中没有独立的 QuaRotModifier**，统一用 `SpinQuantModifier`，通过 `learnable` 参数切换模式。

---

## 2. 四种旋转 (R1-R4)

### 2.1 旋转概览

| 旋转 | 论文名 | 类型 | 位置 | block_size | 运行时开销 |
|---|---|---|---|---|---|
| **R1** | $R_{res}$ | 离线 (offline) | 全连接层间 (残差流) | hidden_size | ✅ 全部可折叠，零开销 |
| **R2** | $R_{ov}^h$ | 离线 (offline) | 注意力头内 (V↔O) | head_dim | ✅ 全部可折叠，零开销 |
| **R3** | $R_{qk}$ | 在线 (online) | Q/K KV-cache | head_dim | ❌ 全部不可折叠，需运行时 hook |
| **R4** | $R_{down}$ | 半在线 (hybrid) | FFN down_proj 激活 | 自定义 (常用 256) | ⚠️ `weight_input` 可折叠，`input` 不可折叠 |

- **离线旋转** (R1/R2): 所有 location 均为 `weight_*`，可在 `on_end()` 时通过 `_fold_transforms_into_weights()` 永久融入权重，部署时零开销
- **在线旋转** (R3): 所有 location 均为 `q_attn`/`k_cache`，运行时通过 attention hook 执行，不可折叠
- **混合旋转** (R4): `weight_input` 部分可离线折叠进 `down_proj` 权重；`input` 部分必须在线通过 pre-hook 旋转激活

> **详见 [§2.6 旋转可折叠性分析](#26-旋转可折叠性分析——数学推导与代码路径)**

### 2.2 R1 — 残差流旋转

**目的**: 减少跨层量化误差传播

```
embed_tokens ──→ ×R ──→ [Layer 0] ──→ [Layer 1] ──→ ... ──→ ×R^{-1} ──→ lm_head
```

**代码** (`_create_r1_scheme`):
```python
TransformScheme(
    head_dim=transform_block_size_R1,   # = hidden_size (如 1536)
    apply=[
        # R 乘在这些层的 weight_output 维度
        TransformArgs(
            targets=[embedding, mm_proj, attn_o, mlp_out],
            location="weight_output",
        ),
        # R^{-1} 乘在这些层的 weight_input 维度
        TransformArgs(
            targets=[attn_q, attn_k, attn_v, mlp_in, lm_head],
            location="weight_input",
            inverse=True,
        ),
    ],
)
```

**R1 涉及的层**:

| 端 | location | 涉及模块 | 说明 |
|---|---|---|---|
| 输出端 (×R) | `weight_output` | `embed_tokens`, `mm_proj`, `attn_o`, `down_proj` | 这些层的输出被旋转 |
| 输入端 (×R⁻¹) | `weight_input` | `q/k/v_proj`, `up/gate_proj`, `lm_head` | 这些层的输入被逆旋转 |

**block_size**: 必须设为 `hidden_size`（如 Qwen3-VL 2B = 1536），否则维度不匹配。

### 2.3 R2 — 注意力头内旋转

**目的**: 减少 V→O 投影间的头内量化误差

```
v_proj ──→ ×R_head ──→ [attention] ──→ ×R_head^{-1} ──→ o_proj
```

**代码** (`_create_r2_scheme`):
```python
TransformScheme(
    block_wise=True,                    # 按 head 维度分块
    head_dim=head_dim,                  # = hidden_size // num_heads (如 128)
    apply=[
        TransformArgs(targets=[attn_v], location="weight_output"),
        TransformArgs(targets=[attn_o], location="weight_input", inverse=True),
    ],
)
```

**`block_wise=True`**: R2 是 block-diagonal 矩阵，每个 attention head 有独立的旋转。

### 2.4 R3 — Q/K KV-cache 旋转

**目的**: 运行时旋转 Q/K，减少 KV-cache 量化误差

```
q_proj ──→ RoPE ──→ ×R3 ──→ [attention score]
k_proj ──→ RoPE ──→ ×R3 ──→ [KV cache]
```

> **关键**: 如论文 Figure 1(b) 所示，R3 位于 **RoPE 之后**、Softmax 之前。
> RoPE 是位置相关的动态运算，R3 无法穿透 RoPE 折叠到 W_q/W_k 权重中。

**代码** (`_create_r3_scheme`):
```python
TransformScheme(
    block_wise=True,
    head_dim=head_dim,
    apply=[
        TransformArgs(targets=[attn], location="q_attn"),
        TransformArgs(targets=[attn], location="k_cache"),
    ],
)
```

**注意**: R3 是在线旋转，不能融入权重，需要运行时计算。原因有三：
1. R3 在 RoPE **之后**，不能穿透位置编码折叠到线性层权重
2. R3 逐 head 旋转（`block_wise=True`），必须在 multi-head reshape 之后执行
3. K 的旋转还需作用于 KV-cache 中已缓存的 key（动态数据）

### 2.5 R4 — FFN down_proj 激活旋转

**目的**: 旋转 down_proj 输入激活，减少 FFN 内的量化误差

```
[up_proj · gate_proj 输出] ──→ ×R_down ──→ ×R_down^{-1} ──→ down_proj
                              (激活旋转)     (权重逆旋转)
```

**代码** (`_create_r4_scheme`):
```python
TransformScheme(
    block_wise=True,
    head_dim=transform_block_size_R4,   # 常用 256
    apply=[
        TransformArgs(targets=[mlp_out], location="input"),           # 激活旋转
        TransformArgs(targets=[mlp_out], location="weight_input", inverse=True),  # 权重逆旋转
    ],
)
```

**block_size_R4**: 常用 256，比 hidden_size 小，以减少运行时开销。`weight_input` 部分可融入权重，`input` 部分需运行时 hook。

### 2.6 旋转可折叠性分析——数学推导与代码路径

#### 2.6.1 核心判断逻辑：`TransformLocation.is_online()`

可折叠性的根本判断在 `compressed_tensors` 的 `TransformLocation` 枚举中：

```python
# compressed_tensors/transform/transform_args.py
class TransformLocation(str, Enum):
    INPUT = "input"               # 在线 → forward pre-hook
    WEIGHT_INPUT = "weight_input" # 离线 → 可折叠进权重
    WEIGHT_OUTPUT = "weight_output" # 离线 → 可折叠进权重
    OUTPUT = "output"             # 在线 → forward hook
    K_CACHE = "k_cache"           # 在线 → kv-cache hook
    Q_ATTN = "q_attn"             # 在线 → attention hook

    def is_online(self) -> bool:
        return self not in (
            TransformLocation.WEIGHT_INPUT,
            TransformLocation.WEIGHT_OUTPUT,
        )
```

**规则**: 只有 `WEIGHT_INPUT` 和 `WEIGHT_OUTPUT` 返回 `is_online() == False`（离线/可折叠），其余全部为在线/不可折叠。

#### 2.6.2 离线 vs 在线的代码分支

旋转应用的核心分发逻辑在 `compressed_tensors/transform/factory/base.py` 的 `_apply_to_module` 方法中，根据 `location` 走不同分支：

**分支 A — `WEIGHT_INPUT` / `WEIGHT_OUTPUT`（可折叠）**:
```python
elif args.location in (TransformLocation.WEIGHT_INPUT, TransformLocation.WEIGHT_OUTPUT):
    # 直接乘到权重张量里
    with torch.no_grad(), align_module_device(module):
        if not is_parametrized(module, "weight"):
            update_offload_parameter(module, "weight", transform(module.weight))

    if self.scheme.requires_grad:  # 训练模式 (SpinQuant learnable=True)
        # 注册 torch parametrization，反向传播可更新 transform 参数
        P.register_parametrization(module, "weight", transform)
    else:  # 推理模式 (QuaRot learnable=False)
        # transform 已直接乘入权重，删除 transform 对象 → 零运行时开销
        delattr(module, transform_name)
```

**分支 B — `INPUT`（不可折叠，forward pre-hook）**:
```python
if args.location == TransformLocation.INPUT:
    def input_hook(_, args):
        input = args[0]
        return transform(input)   # 每次 forward 都要计算！
    module.register_forward_pre_hook(input_hook, prepend=True)
```

**分支 C — `Q_ATTN`（不可折叠，attention query hook）**:
```python
elif args.location == TransformLocation.Q_ATTN:
    def query_hook(_, query_states):
        return transform(query_states)
    initialize_hooked_attention(model, module)
    register_query_hook(module, query_hook)
```

**分支 D — `K_CACHE`（不可折叠，kv-cache key hook）**:
```python
elif args.location == TransformLocation.K_CACHE:
    def key_hook(_, key_states):
        return transform(key_states)
    initialize_hooked_kv_cache(model, module)
    register_key_hook(module, key_hook)
```

#### 2.6.3 折叠触发路径

```python
# SpinQuantModifier.on_end()
def on_end(self, state, event, **kwargs):
    if self.do_fold:
        self._fold_transforms_into_weights(state.model)  # 只处理 parametrize 的部分

# _fold_transforms_into_weights → replace_parametrizations_to_weights
@torch.no_grad()
def replace_parametrizations_to_weights(model, skip_weights_folding=False):
    for _, module in model.named_modules():
        if is_parametrized(module):
            for key in list(module.parametrizations.keys()):
                remove_parametrizations(
                    module, key,
                    leave_parametrized=not skip_weights_folding  # True = 保留旋转后的值
                )
```

`remove_parametrizations(module, "weight", leave_parametrized=True)` 的语义：
- 移除 parametrization 链
- 将 `transform(W_original)` 的**计算结果**写回 `module.weight`
- 旋转**永久融入**权重，部署时 module 就是一个普通 Linear，无任何额外 hook

#### 2.6.4 逐旋转数学论证

##### R1 — 全部可折叠 ✅

R1 位于残差流，旋转相邻层的权重即可：

$$y = xW_{out} \Rightarrow y' = x(W_{out} \cdot R) = xW'_{out}$$
$$z = y'W_{in} \Rightarrow z = (yR)W_{in} = y(R \cdot W_{in})$$

- `weight_output`：$W'_{out} = W_{out} \cdot R$ → 预乘进 embed_tokens/attn_o/down_proj/mm_proj 的权重
- `weight_input`：$W'_{in} = R^{-1} \cdot W_{in}$ → 预乘进 q/k/v_proj, up/gate_proj, lm_head 的权重

两端都是纯权重操作，`is_online() == False`，通过 parametrization → `remove_parametrizations` 折叠。

##### R2 — 全部可折叠 ✅

R2 作用在每个 attention head 内部的 V→O 路径：

$$V' = V \cdot R_{head}, \quad O'_{in} = R_{head}^{-1} \cdot O_{in}$$

- `weight_output` on `v_proj`：$W_V' = W_V \cdot R_{head}$
- `weight_input` on `o_proj`：$W_O' = R_{head}^{-1} \cdot W_O$

同样都是 `WEIGHT_*` location，`block_wise=True` 表示逐 head 应用独立旋转块。

##### R3 — 全部不可折叠 ❌

R3 旋转 Q/K 以减少 KV-cache 量化误差：

$$Q' = Q \cdot R_{qk}, \quad K' = K \cdot R_{qk}$$

注意力分数不变（正交不变性）：$(Q \cdot R)(K \cdot R)^T = QRR^TK^T = QK^T$

**为什么不能折叠到 q_proj/k_proj 的权重里？**

R3 不可折叠有三个层面的原因：

1. **RoPE 阻隔**（最根本）: 如论文 Figure 1(b) 所示，数据流为 `W_q → RoPE → R3`。RoPE 是位置相关的动态运算（$q \cdot e^{i\theta_t}$），R3 无法穿透 RoPE 折叠到 W_q/W_k 权重中。即使不考虑 per-head 问题，RoPE 的存在就决定了 R3 必须在线计算。

2. **Per-head 独立旋转**: R3 逐 head 旋转（`block_wise=True`），必须在 multi-head reshape 之后执行。而 `q_proj`/`k_proj` 的权重是所有 head 合并的完整矩阵，在 GQA 中 K 的 num_heads ≠ Q 的 num_heads，维度不一致。

3. **KV-cache 动态数据**: R3 还需作用于 KV-cache 中已缓存的 key（`k_cache` location），这是运行时动态生成的数据，无法预计算。

代码中的体现：`q_attn` 和 `k_cache` 都走 hook 分支（`register_query_hook` / `register_key_hook`），在 attention 计算内部执行。

##### R4 — 部分可折叠 ⚠️

R4 有两个 TransformArgs，分别走不同分支：

| TransformArgs | location | `is_online()` | 折叠 | 机制 |
|---|---|---|---|---|
| 激活旋转 $h' = h \cdot R$ | `input` | `True` | ❌ | `register_forward_pre_hook` |
| 权重补偿 $W' = R^{-1} \cdot W$ | `weight_input` | `False` | ✅ | parametrization → fold |

数学推导：

$$\text{down\_proj}(h) = h \cdot W_{down}$$

插入旋转后：

$$= (h \cdot R_{blk}) \cdot (R_{blk}^{-1} \cdot W_{down}) = h \cdot W_{down}$$

其中：
- $h \cdot R_{blk}$：**激活旋转**，$h$ 是运行时动态值（up_proj 和 gate_proj 的乘积），每次 forward 不同，**不可能预计算**
- $R_{blk}^{-1} \cdot W_{down}$：**权重补偿**，$W_{down}$ 和 $R_{blk}$ 都是静态的，可以一次性预乘进 `down_proj.weight`

因此 R4 部署后，`down_proj.weight` 已吸收了 $R^{-1}$（零开销），但每次 forward 仍需对输入激活做一次 $h \cdot R_{blk}$ 旋转（通过 pre-hook）。

#### 2.6.5 总结

```
旋转可折叠性判定流程:

  TransformArgs.location
      │
      ├── weight_input / weight_output
      │       │
      │       └── is_online() == False → 离线
      │               │
      │               ├── learnable=False (QuaRot): transform 直接乘入权重并删除
      │               └── learnable=True (SpinQuant): 注册 parametrization，训练后 fold
      │
      └── input / output / q_attn / k_cache
              │
              └── is_online() == True → 在线
                      │
                      └── 注册 forward hook，运行时每次 forward 执行旋转
                          (不可折叠，部署时保留 hook)
```

---

## 3. 完整生命周期

### 3.1 Lifecycle 概览

```
SpinQuantModifier 生命周期:

  on_initialize(state)
    ├── infer_mapping_from_model()     ← 按 model.__class__.__name__ 查 REGISTRY
    ├── infer_norm_mapping_from_model()
    ├── _create_r1_scheme()            ← 构建 R1 TransformScheme
    ├── _create_r2_scheme()            ← 构建 R2 TransformScheme
    ├── _create_r3_scheme()            ← 构建 R3 TransformScheme (可选)
    └── _create_r4_scheme()            ← 构建 R4 TransformScheme (可选)
        ↓
  on_start(state, event)
    ├── untie_word_embeddings()        ← 解绑 embedding 和 lm_head (避免互相影响)
    ├── _center_embeddings()           ← embed_tokens 均值归零 (可选, backe_mean=True)
    ├── _bake_mean_into_fc()           ← attn_o/mlp_out 均值归零 (可选)
    ├── _fuse_norms()                  ← RMSNorm 权重融入相邻 Linear
    └── apply_transform_config()       ← 插入旋转矩阵到模型
        ↓
  [校准数据 forward pass]              ← GPTQ 校准 (QuaRot 模式下)
  [或 FSDP 训练]                       ← OSTQuant 模式下
        ↓
  on_end(state, event)
    └── _fold_transforms_into_weights()  ← 将离线旋转融入权重 (do_fold=True)
        ↓
  on_finalize(state)
```

### 3.2 映射注册机制

```python
# 1. 用户在脚本顶层注册映射
mappings.SPINQUANT_MAPPING_REGISTRY["Qwen3VLForConditionalGeneration"] = SpinQuantMapping(...)
norm_mappings.NORM_MAPPING_REGISTRY["Qwen3VLForConditionalGeneration"] = [...]

# 2. oneshot() 内部触发 on_initialize()
self.mappings = infer_mapping_from_model(state.model)
# → architecture = model.__class__.__name__  # "Qwen3VLForConditionalGeneration"
# → return SPINQUANT_MAPPING_REGISTRY.get(architecture, _default_mappings)

# 3. 查到映射后构建旋转 scheme
```

**内置注册**:
| 模型 | 预注册 |
|---|---|
| `LlamaForCausalLM` | ✅ (默认映射) |
| 其他模型 | 需用户手动注册 |

### 3.3 _fuse_norms 详解

Norm fusing 是 SpinQuant 的关键前置步骤：将 RMSNorm 的缩放权重融入后续 Linear 层的权重中。

**为什么需要 fuse**: 正交旋转可以与归一化运算交换 ($R \cdot \text{norm}(x) = \text{norm}(R \cdot x)$)，但 **不能** 与缩放运算交换。因此必须先将 norm 的 scale 权重融入 Linear，使 norm 变为"纯归一化"。

```python
# 融入过程:
new_weight = linear.weight * norm.weight   # Linear 权重吸收 norm scale
norm.weight = all_ones                      # norm 变为纯归一化 (scale=1)
```

**Norm 映射** 定义了哪些 norm 的权重融入哪些 Linear:

| Norm | 融入的 Linear |
|---|---|
| `input_layernorm` | `q_proj`, `k_proj`, `v_proj` |
| `post_attention_layernorm` | `up_proj`, `gate_proj` |
| `model.language_model.norm` (最后一层) | `lm_head` |

---

## 4. SpinQuantMapping 字段详解

```python
class SpinQuantMapping(BaseModel):
    mm_proj: List[str]       # 多模态投影层 (如 visual.merger.linear_fc2)
    embedding: List[str]     # 嵌入层 (embed_tokens) — R1 旋转用，不量化
    attn: str                # 注意力模块 (self_attn) — R3 目标
    attn_q: str              # Q 投影 — R1 输入端
    attn_k: str              # K 投影 — R1 输入端
    attn_v: str              # V 投影 — R1 输入端 + R2 输出端
    attn_o: str              # O 投影 — R1 输出端 + R2 输入端
    attn_head_dim: int       # 注意力头维度 (head_dim)
    mlp_in: List[str]        # FFN 输入 (up_proj, gate_proj) — R1 输入端
    mlp_out: List[str]       # FFN 输出 (down_proj) — R1 输出端 + R4 目标
    lm_head: List[str]       # LM 头 — R1 输入端
```

### 各旋转对映射字段的引用

| 旋转 | weight_output (×R) | weight_input (×R⁻¹) |
|---|---|---|
| **R1** | embedding, mm_proj, attn_o, mlp_out | attn_q, attn_k, attn_v, mlp_in, lm_head |
| **R2** | attn_v | attn_o |
| **R3** | — | — (q_attn, k_cache 特殊 location) |
| **R4** | — (input 激活) | mlp_out |

---

## 5. 旋转在模型中的位置（数据流视图）

> 对应论文 Figure 1。实线框 = 可合并(Mergeable)旋转 (R1/R2)，虚线框 = 在线(Online)旋转 (R3/R4)

### 5.1 (a) 残差流 — R1

```
Embed(W_e) ──→ [R1] ──→ [R1⁻¹]──→ Multi-Head Attn ──→ [R1] ──→ [R1⁻¹]──→ FFN ──→ [R1] ──→ ... ──→ [R1⁻¹]──→ Output(W_head)
                         ↑ 残差流携带旋转后激活 X@R1          ↑ 残差流携带 X@R1
```

R1 在残差流中成对出现，相邻层的 R1/R1⁻¹ 可分别合并进两侧权重。

### 5.2 (b) Attention 块内部 — R1⁻¹, R2, R3

```
       ┌──→ [R1⁻¹·W_q] ──→ RoPE ──→ [R3]* ──→ ·───────┐
       │                                                 │ Softmax
X@R1 ──┼──→ [R1⁻¹·W_k] ──→ RoPE ──→ [R3]* ──→ ·───────┘
       │                                          ↑ KV-cache quantization
       └──→ [R1⁻¹·W_v·R2] ──→ ·──→ [R2⁻¹·W_o·R1] ──→ + residual ──→ X@R1
```

- `[R1⁻¹·W_q]`: R1⁻¹ 合并进 q_proj 权重（Mergeable）
- `[R1⁻¹·W_k]`: R1⁻¹ 合并进 k_proj 权重（Mergeable）
- `[R1⁻¹·W_v·R2]`: R1⁻¹ 和 R2 **同时**合并进 v_proj 权重（Mergeable）
- `[R2⁻¹·W_o·R1]`: R2⁻¹ 和 R1 **同时**合并进 o_proj 权重（Mergeable）
- `[R3]*`: 虚线 = Online，在 RoPE **之后**逐 head 旋转 Q/K（不可合并）

### 5.3 (c) FFN 块内部 — R1⁻¹, R4

```
       ┌──→ [R1⁻¹·W_up]   ──→ ──┐
X@R1 ──┤                         ⊗ ──→ [R4]* ──→ [R4⁻¹·W_down·R1] ──→ + residual ──→ X@R1
       └──→ [R1⁻¹·W_gate] ──→ Swish ┘
```

- `[R1⁻¹·W_up]`: R1⁻¹ 合并进 up_proj 权重（Mergeable）
- `[R1⁻¹·W_gate]`: R1⁻¹ 合并进 gate_proj 权重（Mergeable）
- `[R4]*`: 虚线 = Online，旋转 SiLU(gate)⊗up 的激活输出（不可合并）
- `[R4⁻¹·W_down·R1]`: R4⁻¹ 和 R1 **同时**合并进 down_proj 权重（Mergeable）

### 5.4 折叠后各权重的合并结果

折叠完成后，每个权重矩阵实际吸收的旋转如下：

| 权重矩阵 | 合并后等效权重 | 吸收的旋转 |
|---|---|---|
| `embed_tokens` | $W_e \cdot R_1$ | R1 output |
| `q_proj` | $R_1^{-1} \cdot W_q$ | R1 input |
| `k_proj` | $R_1^{-1} \cdot W_k$ | R1 input |
| `v_proj` | $R_1^{-1} \cdot W_v \cdot R_2$ | R1 input + R2 output |
| `o_proj` | $R_2^{-1} \cdot W_o \cdot R_1$ | R2 input + R1 output |
| `up_proj` | $R_1^{-1} \cdot W_{up}$ | R1 input |
| `gate_proj` | $R_1^{-1} \cdot W_{gate}$ | R1 input |
| `down_proj` | $R_4^{-1} \cdot W_{down} \cdot R_1$ | R4 input + R1 output |
| `lm_head` | $R_1^{-1} \cdot W_{head}$ | R1 input |
| `mm_proj` | $W_{mm} \cdot R_1$ | R1 output |

> **注意**: v_proj、o_proj、down_proj 各自同时吸收了**两个不同旋转**矩阵。这在代码中体现为同一个 module 上先后注册了两次 parametrization（或两次 weight 更新），顺序由 R1/R2/R4 scheme 的 apply 顺序决定。

---

## 6. 配置参数一览

### SpinQuantModifier 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `rotations` | `["R1", "R2"]` | 启用的旋转列表 |
| `transform_type` | `"hadamard"` | 旋转类型: `hadamard` / `random-hadamard` / `random-matrix` / `identity` |
| `learnable` | `False` | 是否可学习 (False=QuaRot, True=SpinQuant) |
| `randomize` | `False` | 每个应用点是否用独立随机矩阵 (暂未支持) |
| `precision` | `torch.float64` | 旋转计算精度 |
| `transform_block_size_R1` | `None` → hidden_size | R1 旋转块大小 |
| `transform_block_size_R2` | `None` → head_dim | R2 旋转块大小 |
| `transform_block_size_R3` | `None` → head_dim | R3 旋转块大小 |
| `transform_block_size_R4` | `None` | R4 旋转块大小 (常用 256) |
| `do_fold` | `True` | 是否在 on_end 时将旋转融入权重 |
| `backe_mean` | `False` | 是否对 embedding 和 FC 做均值归零 |
| `sequential_onload` | `False` | 是否逐层加载 (大模型省显存) |

### transform_type 对比

| 类型 | 矩阵构造 | 尺寸限制 | 性能开销 | 适用场景 |
|---|---|---|---|---|
| `hadamard` | 固定 Hadamard | 2 的幂次 | 最低 | QuaRot (非学习) |
| `random-hadamard` | 随机 Hadamard | 2 的幂次 | 中等 | SpinQuant / OSTQuant (可学习) |
| `random-matrix` | 随机正交矩阵 | 任意 | 最高 | 非常规尺寸模型 |
| `identity` | 单位矩阵 | 任意 | 无 | 调试/基线 |

---

## 7. 与量化 Modifier 配合

### 7.1 QuaRot + GPTQ (YAML 配置)

```yaml
quant_stage:
  quant_modifiers:
    SpinQuantModifier:              # learnable 默认 False → QuaRot
      rotations: ["R1", "R2", "R4"]
      transform_block_size_R1: 1536
      transform_block_size_R4: 256
      transform_type: "hadamard"
    GPTQModifier:
      ignore: ["re:.*lm_head", "re:.*visual.*"]
      config_groups:
        group_0:
          weights: {num_bits: 4, type: int, symmetric: true, strategy: channel}
          targets: ["re:.*q_proj$", "re:.*k_proj$", ...]
```

**执行顺序** (在 `oneshot()` 中):
1. `SpinQuantModifier.on_initialize()` — 构建旋转 scheme
2. `SpinQuantModifier.on_start()` — fuse norms + 插入旋转
3. `GPTQModifier` 校准 — 在旋转后的模型上做 GPTQ
4. `SpinQuantModifier.on_end()` — 旋转融入权重
5. 保存模型

### 7.2 OSTQuant 训练 + PostQuant

```
Phase 1: SpinQuantModifier(learnable=True) + QuantizationModifier(ste=True)
    → FSDP 训练 R1+R2 (KL 蒸馏)
    → fold R1+R2 into weights
    → 保存 transformed model

Phase 2: SpinQuantModifier(rotations=["R4"]) + GPTQModifier
    → 在 transformed model 上 oneshot
    → 保存最终量化模型
```

### 7.3 SpinQuant + 其他量化方法

SpinQuantModifier 可与任意量化 Modifier 组合:
- `SpinQuantModifier` + `GPTQModifier` — GPTQ 量化
- `SpinQuantModifier` + `QuantizationModifier` — RTN / STE 量化
- `SpinQuantModifier` + `AWQModifier` — AWQ 量化

---

## 8. 关键实现细节

### 8.1 center_embeddings — 嵌入均值归零

```python
def center_embeddings(embedding):
    weight = embedding.weight.to(float64)
    new_weight = weight - weight.mean(dim=-1, keepdim=True)  # 每行减均值
    embedding.weight = new_weight
```

**目的**: 减小嵌入层输出的动态范围，使后续旋转更有效。仅在 `backe_mean=True` 时启用。

### 8.2 back_mean_into_fc — FC 输出均值归零

```python
def back_mean_into_fc(linear):
    new_weight = linear.weight - linear.weight.mean(dim=-2, keepdim=True)  # 列均值归零
    linear.weight = new_weight
```

**目的**: 对 `attn_o`, `mlp_out`, `mm_proj` 的输出做均值归零，与 `center_embeddings` 配合。

### 8.3 untie_word_embeddings — 解绑嵌入

```python
untie_word_embeddings(model)
```

许多 HF 模型默认 `embed_tokens.weight` 和 `lm_head.weight` 共享同一张量 (tied)。SpinQuant 需要对它们分别施加不同方向的旋转 (R1 output vs R1 input inverse)，所以必须先解绑。

### 8.4 replace_parametrizations_to_weights — 融入权重

`on_end()` → `_fold_transforms_into_weights()` → `replace_parametrizations_to_weights()` 的完整逻辑和数学推导详见 [§2.6 旋转可折叠性分析](#26-旋转可折叠性分析——数学推导与代码路径)。

核心语义：
```python
# 融入前: module.weight → parametrization(original)  [每次 forward 计算 transform(W)]
# 融入后: module.weight = transform(W) 的最终值      [parametrization 被移除，零开销]
```

---

## 9. 代码文件结构

```
src/llmcompressor/modifiers/transform/spinquant/
├── __init__.py
├── base.py           # SpinQuantModifier 主类 (生命周期 + R1-R4 scheme 构建)
├── mappings.py       # SpinQuantMapping 数据类 + REGISTRY + infer_mapping_from_model()
└── norm_mappings.py  # NormMapping 数据类 + REGISTRY + infer_norm_mapping_from_model()

src/llmcompressor/modeling/
├── fuse.py           # center_embeddings / back_mean_into_fc / fuse_norm_linears
└── replace.py        # replace_ln_to_rmsnorm / replace_parametrizations_to_weights

# compressed-tensors (外部依赖):
compressed_tensors/transform/
├── TransformScheme   # 旋转方案 (type, head_dim, block_wise, requires_grad, apply)
├── TransformArgs     # 旋转参数 (targets, location, inverse)
├── TransformConfig   # config_groups: Dict[str, TransformScheme]
└── apply_transform_config()  # 实际将旋转插入模型
```

---

## 10. Qwen3-VL 映射注册示例

```python
from llmcompressor.modifiers.transform.spinquant import mappings, norm_mappings

# SpinQuant 映射
mappings.SPINQUANT_MAPPING_REGISTRY["Qwen3VLForConditionalGeneration"] = (
    mappings.SpinQuantMapping(
        mm_proj=[r"re:.*visual\.merger.*linear_fc2$"],
        embedding="re:.*embed_tokens$",
        attn="re:.*self_attn$",
        attn_q="re:.*language_model.*q_proj$",
        attn_k="re:.*language_model.*k_proj$",
        attn_v="re:.*language_model.*v_proj$",
        attn_o="re:.*language_model.*o_proj$",
        mlp_in=[r"re:.*language_model.*mlp\.up_proj$", r"re:.*language_model.*mlp\.gate_proj$"],
        mlp_out=[r"re:.*language_model.*mlp\.down_proj$"],
        lm_head="lm_head",
    )
)

# Norm 映射
norm_mappings.NORM_MAPPING_REGISTRY["Qwen3VLForConditionalGeneration"] = [
    norm_mappings.NormMapping(
        norm="re:.*language_model.*input_layernorm$",
        linears=["re:.*language_model.*q_proj$", "re:.*language_model.*k_proj$", "re:.*language_model.*v_proj$"],
    ),
    norm_mappings.NormMapping(
        norm="re:.*language_model.*post_attention_layernorm$",
        linears=[r"re:.*language_model.*mlp\.up_proj$", r"re:.*language_model.*mlp\.gate_proj$"],
    ),
    norm_mappings.NormMapping(
        norm="model.language_model.norm",
        linears=["lm_head"],
    ),
]
```

---

## 11. 官方例子

### 11.1 最简 SpinQuant + 量化 (Llama)

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform import SpinQuantModifier

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")

recipe = [
    SpinQuantModifier(
        rotations=["R1", "R2", "R4"],
        transform_block_size=128,
        transform_type="hadamard",
    ),
    QuantizationModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]),
]

oneshot(model=model, recipe=recipe, pipeline="datafree")
model.save_pretrained(SAVE_DIR, save_compressed=True)
```

**注意**: Llama 已有默认映射 (`LlamaForCausalLM` 预注册在 REGISTRY)，无需手动注册。

### 11.2 自定义模型 (需注册映射)

对于 REGISTRY 中没有的模型（如 Qwen3-VL），需在 `oneshot()` 之前手动注册映射。见第 10 节示例。
