# llm-compressor 架构概要

> **源码根**: `src/llmcompressor/`
> **版本**: 基于当前 `hh_exp` 分支
> **定位**: 基于 PyTorch + HuggingFace Transformers 的 LLM 压缩库，支持后训练量化 (PTQ)、稀疏化、以及训练感知压缩

---

## 1. 顶层架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                           用户入口                                   │
│        oneshot()  /  Oneshot  /  model_free_ptq  /  train           │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │      Args 解析层         │
                    │  ModelArgs / DatasetArgs │
                    │  RecipeArgs / parse_args │
                    └────────────┬────────────┘
                                 │
              ┌──────────────────▼──────────────────┐
              │        Core 编排层                    │
              │  CompressionSession                  │
              │  ├─ CompressionLifecycle              │
              │  │  ├─ Recipe (modifiers 列表)       │
              │  │  └─ EventType 事件驱动             │
              │  └─ State (model/data/hardware)      │
              └──────────────────┬──────────────────┘
                                 │
              ┌──────────────────▼──────────────────┐
              │        Pipeline 执行层               │
              │  Sequential / Basic / DataFree /     │
              │  Independent / ModelFree             │
              └──────────────────┬──────────────────┘
                                 │
     ┌───────────────┬───────────▼──────────┬───────────────┐
     │               │                      │               │
┌────▼────┐   ┌──────▼──────┐   ┌──────────▼──┐   ┌───────▼──────┐
│ Modifier │   │  Observer   │   │  Modeling   │   │   Datasets   │
│ (算法层) │   │ (观测器)    │   │  (模型工具)  │   │  (数据加载)   │
└─────────┘   └─────────────┘   └─────────────┘   └──────────────┘
```

---

## 2. 模块职责

### 2.1 入口层 (`entrypoints/`)

| 入口 | 说明 |
|---|---|
| `oneshot()` / `Oneshot` | **主入口**。后训练压缩：加载模型 → 校准 → 保存 |
| `model_free_ptq` | 无需加载完整模型的 PTQ（直接操作权重文件） |
| `train` (`train/`) | 训练感知压缩（FSDP 分布式训练支持） |

**`Oneshot` 的完整流程**:

```
Oneshot.__init__()
  ├─ parse_args(**kwargs)           → ModelArgs, DatasetArgs, RecipeArgs
  └─ pre_process(model_args, ...)   → 加载模型/processor, untie embeddings, patch model

Oneshot.__call__()
  ├─ get_calibration_dataloader()   → 构建校准数据 DataLoader
  ├─ apply_recipe_modifiers()
  │    ├─ session = active_session()
  │    ├─ session.initialize(model, recipe, calib_data, ...)
  │    ├─ pipeline = CalibrationPipeline.from_modifiers(modifiers)
  │    ├─ pipeline(model, dataloader, dataset_args)    ← 核心执行
  │    └─ session.finalize()
  └─ post_process()                 → 保存模型/processor/config
```

### 2.2 参数解析层 (`args/`)

| 文件 | 职责 |
|---|---|
| `model_arguments.py` | 模型加载参数 (model_name_or_path, dtype, device_map, trust_remote_code, ...) |
| `dataset_arguments.py` | 数据集参数 (dataset, num_calibration_samples, max_seq_length, pipeline, ...) |
| `recipe_arguments.py` | Recipe 参数 (recipe YAML/JSON 路径, stage, recipe_args) |
| `utils.py` | `parse_args()` 统一解析入口 |

### 2.3 Core 编排层 (`core/`)

核心是 **Session → Lifecycle → State → Event** 的事件驱动架构：

```
CompressionSession
  ├─ lifecycle: CompressionLifecycle
  │    ├─ recipe: Recipe
  │    │    └─ modifiers: List[Modifier]   ← 从 recipe 解析出的所有 modifier
  │    ├─ state: State
  │    └─ event(EventType, global_step)    ← 事件分发
  └─ 对外方法:
       initialize() → lifecycle.initialize()  → 逐个 mod.initialize(state)
       finalize()   → lifecycle.finalize()    → 逐个 mod.finalize(state)
       event()      → lifecycle.event()       → 逐个 mod.update_event(state, event)
```

#### State 数据结构

```python
@dataclass
class State:
    model: Any                    # 被压缩的模型
    teacher_model: Any            # 蒸馏用教师模型 (可选)
    optimizer: Any                # 优化器 (训练模式)
    data: Data                    # 训练/校准/测试数据
    hardware: Hardware            # 设备/分布式信息
    start: float                  # 开始步数
    recipe: Recipe                # 当前 recipe
    sequential_targets: List[str] # sequential pipeline 分层目标
    calib_data: DataLoader        # 校准 dataloader
```

#### 事件类型 (EventType)

```python
class EventType(Enum):
    # 全局生命周期
    INITIALIZE              # 初始化
    FINALIZE                # 结束

    # Batch 生命周期 (训练)
    BATCH_START             # batch 开始
    LOSS_CALCULATED         # loss 计算完成
    BATCH_END               # batch 结束

    # 校准生命周期
    CALIBRATION_EPOCH_START # 校准 epoch 开始
    SEQUENTIAL_EPOCH_END    # sequential pipeline 一层校准完成
    CALIBRATION_EPOCH_END   # 校准 epoch 结束

    # 优化器生命周期 (训练)
    OPTIM_PRE_STEP          # 优化器 step 之前
    OPTIM_POST_STEP         # 优化器 step 之后
```

### 2.4 Pipeline 执行层 (`pipelines/`)

Pipeline 负责：**数据如何流过模型** + **何时触发生命周期事件**

| Pipeline | 注册名 | 适用场景 | 工作方式 |
|---|---|---|---|
| **SequentialPipeline** | `sequential` | GPTQ/SparseGPT 等需要逐层校准的算法 | 将模型切分为子图，逐子图前向，缓存中间激活 |
| **BasicPipeline** | `basic` | 简单全模型前向校准 | 整个模型 forward，触发 hooks |
| **DataFreePipeline** | `datafree` | 无需校准数据的量化 (如纯权重 RTN) | 只触发事件，不跑数据 |
| **IndependentPipeline** | `independent` | 多个 modifier 需独立校准 | 每个 modifier 单独走一轮 pipeline |
| **ModelFreePipeline** | `model_free` | 不加载完整模型的 PTQ | 直接操作权重文件 |

#### Pipeline 自动推断逻辑

```python
CalibrationPipeline.from_modifiers(modifiers, user=None)
  → 如果 user 指定了 pipeline，用 user 指定的
  → 否则自动推断:
      if 只有 1 个 QuantizationModifier 且不需要校准数据:
          return "datafree"
      else:
          return "sequential"     ← 默认
```

#### SequentialPipeline 工作流（最常用）

```
SequentialPipeline.__call__(model, dataloader, dataset_args):
  1. get_sequential_targets()          → 确定分层目标 (如 "Qwen3VLTextDecoderLayer")
  2. trace_subgraphs(model, targets)   → 将模型切分为子图列表
  3. IntermediatesCache.from_dataloader → 构建激活缓存 (首次前向)

  4. for subgraph in subgraphs:        ← 逐子图处理
       LifecycleCallbacks.calibration_epoch_start()
       for batch in dataloader:
           activations = cache.fetch(batch_idx)     ← 从缓存取上一层输出
           output = subgraph.forward(activations)   ← 前向 (触发 modifier hooks)
           cache.update(batch_idx, output)           ← 更新缓存
       LifecycleCallbacks.sequential_epoch_end()    ← 触发 SEQUENTIAL_EPOCH_END
                                                     → modifier 执行权重量化

  5. LifecycleCallbacks.calibration_epoch_end()
```

**IntermediatesCache**: 管理层间激活的缓存系统，支持 CPU offload 节省显存。

### 2.5 Recipe 系统 (`recipe/`)

Recipe 是压缩配置的声明式描述，支持 YAML/JSON/Python list 格式：

```yaml
# 示例 recipe
quant_stage:
  quant_modifiers:
    SpinQuantModifier:
      rotations: ["R1", "R2", "R4"]
    GPTQModifier:
      config_groups:
        group_0:
          weights: {num_bits: 4, ...}
          targets: ["Linear"]
```

```python
class Recipe:
    args: Dict                    # 全局参数
    modifiers: List[Modifier]     # 解析后的 modifier 对象列表

    @classmethod
    def create_instance(path_or_modifiers, target_stage)  # 从文件/列表创建
```

Recipe 解析流程：YAML 文件 → 解析出 modifier 类型和参数 → `ModifierFactory.create()` 实例化 → `Recipe.modifiers` 列表

### 2.6 Modifier 系统 (`modifiers/`)

所有压缩算法都实现为 Modifier，统一生命周期接口：

```python
class Modifier(ModifierInterface, HooksMixin):
    # 生命周期状态
    initialized_: bool
    started_: bool
    ended_: bool
    finalized_: bool

    # 生命周期方法 (子类实现)
    on_initialize(state) → bool         # 初始化：解析配置，准备数据结构
    on_start(state, event)              # 开始：注册 hooks 到模型
    on_event(state, event)              # 事件响应：如 SEQUENTIAL_EPOCH_END 时执行量化
    on_end(state, event)                # 结束：清理 hooks
    on_finalize(state) → bool           # 最终化：保存状态
```

```
生命周期调用顺序:

  initialize()
    └─ on_initialize()       ← 初始化配置
    └─ on_start()            ← 自动触发 (如果 should_start)
         │
  [Pipeline 运行中]
    └─ update_event()
         ├─ on_start()       ← 首次满足条件时
         ├─ on_update()      ← 每个事件
         └─ on_end()         ← 条件结束时
         │
  finalize()
    └─ on_finalize()         ← 最终清理
```

#### Modifier 汇总

| 类别 | Modifier | 说明 |
|---|---|---|
| **量化** | `QuantizationModifier` | 通用量化框架 (RTN/STE/MinMax)，配置量化方案 |
| **量化** | `GPTQModifier` | GPTQ 算法实现，逐层 Hessian 校准量化 |
| **量化** | `AWQModifier` | Activation-Aware Weight Quantization |
| **量化** | `AutoRoundModifier` | AutoRound 量化 |
| **变换** | `SpinQuantModifier` | SpinQuant/QuaRot 旋转变换 (R1-R4) |
| **变换** | `SmoothQuantModifier` | SmoothQuant/OSTQuant 平滑变换 (S1-S4) |
| **变换** | `QuipModifier` | QUIP 变换 |
| **稀疏** | `MagnitudePruningModifier` | 幅度剪枝 |
| **稀疏** | `SparseGPTModifier` | SparseGPT 算法 |
| **稀疏** | `WandaPruningModifier` | WANDA 剪枝 |
| **稀疏** | `ConstantPruningModifier` | 固定稀疏掩码 |
| **混合** | `OBCQModifier` | Optimal Brain Compression (OBCQ) |
| **均衡** | `LogarithmicEqualizationModifier` | 对数均衡化 |

### 2.7 Observer 系统 (`observers/`)

Observer 负责在校准时确定量化参数（scale, zero_point）：

| Observer | 策略 |
|---|---|
| `MinMaxObserver` | 取 min/max 值 |
| `MSEObserver` | 最小化量化 MSE |
| `PercentileObserver` | 按百分位截断 |
| `KLObserver` | 最小化 KL 散度 |
| `LCTObserver` | Learned Clipping Threshold |
| `MovingBaseObserver` | 移动平均基类 |

### 2.8 Modeling 工具 (`modeling/`)

模型操作工具函数：

| 文件 | 功能 |
|---|---|
| `fuse.py` | `center_embeddings`, `back_mean_into_fc`, `fuse_norm_linears` |
| `replace.py` | `replace_ln_to_rmsnorm`, `replace_parametrizations_to_weights` |
| `moe_context.py` | MoE 模型校准上下文管理 |
| `patch/` | 模型 forward patch (用于量化保存) |
| `*_moe.py` | 各 MoE 模型特化支持 (DeepSeek-V3, Qwen3-MoE, GLM-4-MoE, ...) |

### 2.9 Datasets (`datasets/`)

| 函数 | 功能 |
|---|---|
| `get_processed_dataset()` | 加载 + 分割 + tokenize 数据集 |
| `get_calibration_dataloader()` | 构建校准用 DataLoader |

### 2.10 Train (`train/`)

| 文件 | 功能 |
|---|---|
| `fsdp_trainer.py` | 基于 HF Trainer 的 FSDP 训练器 |
| `train_utils.py` | 训练工具函数 (loss, args) |

---

## 3. 完整调用链

### 3.1 oneshot() 全流程

```
用户调用: oneshot(model="...", recipe=[...], dataset="...")
  │
  ▼
parse_args()                          ← 解析为 ModelArgs + DatasetArgs + RecipeArgs
  │
  ▼
pre_process()
  ├─ load model (AutoModelForCausalLM.from_pretrained)
  ├─ load processor (AutoTokenizer / AutoProcessor)
  ├─ untie_word_embeddings()
  └─ patch model for quantized saving
  │
  ▼
get_calibration_dataloader()
  ├─ load dataset (HF datasets / local)
  ├─ tokenize + pad/truncate
  └─ 构建 DataLoader
  │
  ▼
apply_recipe_modifiers()
  ├─ session = active_session()       ← 获取全局 CompressionSession
  ├─ session.reset()
  ├─ session.initialize()
  │    ├─ Recipe.create_instance()    ← 解析 recipe → modifier 对象列表
  │    └─ for mod in modifiers:
  │         mod.initialize(state)     ← 各 modifier 初始化 (配置/scheme/mapping)
  │              └─ mod.on_start()    ← 注册 hooks (如 GPTQ 的 Hessian 收集 hook)
  │
  ├─ pipeline = CalibrationPipeline.from_modifiers(modifiers)
  │    └─ 推断得到 SequentialPipeline (最常见)
  │
  ├─ pipeline(model, dataloader, dataset_args)
  │    ├─ [SequentialPipeline]
  │    │   ├─ trace_subgraphs(model)              ← 切分模型为子图
  │    │   ├─ IntermediatesCache.from_dataloader   ← 初始缓存
  │    │   ├─ for subgraph in subgraphs:
  │    │   │    ├─ CALIBRATION_EPOCH_START 事件
  │    │   │    ├─ for batch in batches:
  │    │   │    │    ├─ fetch 激活 → subgraph forward → update 缓存
  │    │   │    │    └─ (modifier hooks 在 forward 中被触发：收集 Hessian/统计量)
  │    │   │    ├─ SEQUENTIAL_EPOCH_END 事件
  │    │   │    │    └─ GPTQModifier 响应: 用 Hessian 量化当前层权重
  │    │   │    └─ 清理当前子图 hooks
  │    │   └─ CALIBRATION_EPOCH_END 事件
  │    │
  │    └─ [其他 pipeline 类似但更简单]
  │
  └─ session.finalize()
       └─ for mod in modifiers:
            mod.finalize(state)       ← 清理/保存
  │
  ▼
post_process()
  ├─ model.save_pretrained(output_dir, save_compressed=True)
  ├─ processor.save_pretrained(output_dir)
  └─ save config/recipe
```

### 3.2 Modifier 在 Pipeline 中的交互

```
Pipeline                          Modifier (如 GPTQModifier)
  │                                  │
  ├── initialize ──────────────────→ on_initialize: 解析量化配置
  │                                  on_start: 注册 forward hooks
  │                                      │
  │   ┌─ subgraph forward ──────────→  hook 触发: calibrate_module()
  │   │   (校准数据前向)                   收集 Hessian 矩阵
  │   │                                    │
  │   └─ SEQUENTIAL_EPOCH_END ────→  on_event: compress_modules()
  │       (一层校准完成)                    GPTQ 量化当前层权重
  │                                    移除当前层 hooks
  │                                        │
  │   [下一层重复...]                       │
  │                                        │
  ├── CALIBRATION_EPOCH_END ──────→  on_end: 清理收尾
  │                                        │
  └── finalize ───────────────────→  on_finalize: 最终状态保存
```

### 3.3 变换 + 量化的多 Modifier 协作

```
SpinQuantModifier + GPTQModifier 的典型 recipe:

  session.initialize()
    ├─ SpinQuantModifier.on_initialize()   ← 构建 R1/R2/R4 旋转 scheme
    │    └─ SpinQuantModifier.on_start()   ← fuse norms + 插入旋转
    └─ GPTQModifier.on_initialize()        ← 解析量化配置
         └─ GPTQModifier.on_start()        ← 注册 Hessian hooks

  pipeline 运行 (校准)
    └─ 数据在旋转后的模型上流过 → GPTQModifier 收集 Hessian

  SEQUENTIAL_EPOCH_END × N
    └─ GPTQModifier 逐层量化旋转后的权重

  session.finalize()
    ├─ SpinQuantModifier.on_end()          ← 折叠旋转进权重
    └─ GPTQModifier.on_finalize()          ← 清理
```

---

## 4. 关键设计思想

### 4.1 事件驱动 + 生命周期

所有压缩算法通过统一的 **Modifier 生命周期** 接入，Pipeline 通过 **EventType 事件** 驱动 Modifier 的行为。这种设计使得：
- 不同算法可以自由组合（如 SpinQuant + GPTQ）
- Pipeline 不需要知道具体算法细节
- 新算法只需实现 Modifier 接口

### 4.2 逐层 Sequential 执行

对于 GPTQ 等需要 Hessian 校准的算法，SequentialPipeline 将模型切分为子图逐层处理，配合 IntermediatesCache 缓存中间激活，实现：
- 显存高效（只需一层的 Hessian + 激活缓存）
- 准确性好（每层用前一层量化后的输出做校准，误差不累积到后层）

### 4.3 Recipe 声明式配置

用户通过 YAML/Python list 声明式地描述压缩方案，框架自动解析、实例化 Modifier、推断 Pipeline。降低使用门槛。

### 4.4 compressed-tensors 分离

量化的底层实现（TransformScheme、QuantizationConfig 等数据结构）放在 `compressed-tensors` 独立库中，`llm-compressor` 只负责编排和算法逻辑。

---

## 5. 源码目录结构

```
src/llmcompressor/
├── __init__.py              # 顶层导出: oneshot, active_session, ...
├── logger.py                # loguru 日志配置
├── sentinel.py              # 哨兵对象
├── typing.py                # 类型定义
│
├── args/                    # 参数解析
│   ├── model_arguments.py
│   ├── dataset_arguments.py
│   ├── recipe_arguments.py
│   └── utils.py             # parse_args()
│
├── core/                    # 核心编排
│   ├── session.py           # CompressionSession
│   ├── session_functions.py # 全局 session 管理
│   ├── lifecycle.py         # CompressionLifecycle
│   ├── state.py             # State / Data / Hardware
│   ├── model_layer.py       # ModelParameterizedLayer
│   └── events/
│       └── event.py         # EventType / Event
│
├── entrypoints/             # 用户入口
│   ├── oneshot.py           # Oneshot / oneshot()
│   ├── model_free/          # model_free_ptq 无模型量化
│   └── utils.py             # pre_process / post_process
│
├── pipelines/               # 执行管线
│   ├── registry.py          # CalibrationPipeline 基类 + 推断逻辑
│   ├── cache.py             # IntermediatesCache 激活缓存
│   ├── sequential/          # SequentialPipeline (默认/最常用)
│   ├── basic/               # BasicPipeline (简单全模型前向)
│   ├── data_free/           # DataFreePipeline (无数据)
│   └── independent/         # IndependentPipeline (多 modifier 独立校准)
│
├── recipe/                  # Recipe 系统
│   ├── recipe.py            # Recipe 类
│   ├── metadata.py          # 元数据
│   └── utils.py             # 解析工具
│
├── modifiers/               # 压缩算法 (全部实现为 Modifier)
│   ├── modifier.py          # Modifier 基类
│   ├── interface.py         # ModifierInterface 接口
│   ├── factory.py           # ModifierFactory 工厂
│   ├── quantization/        # QuantizationModifier (通用量化)
│   │   ├── quantization/    #   RTN/STE/MinMax 量化实现
│   │   ├── gptq/            #   GPTQ 量化核心
│   │   └── calibration.py   #   校准辅助
│   ├── gptq/                # GPTQModifier
│   ├── awq/                 # AWQModifier
│   ├── autoround/           # AutoRoundModifier
│   ├── transform/           # 变换类 Modifier
│   │   ├── spinquant/       #   SpinQuantModifier (R1-R4 旋转)
│   │   ├── smoothquant/     #   SmoothQuantModifier (S1-S4 平滑)
│   │   └── quip/            #   QuipModifier
│   ├── smoothquant/         # 独立 SmoothQuantModifier (旧版)
│   ├── pruning/             # 稀疏化 Modifier 集合
│   │   ├── magnitude/       #   幅度剪枝
│   │   ├── sparsegpt/       #   SparseGPT
│   │   ├── wanda/           #   WANDA
│   │   └── constant/        #   固定掩码
│   ├── obcq/                # OBCQModifier
│   ├── logarithmic_equalization/ # 对数均衡化
│   └── experimental/        # 实验性 Modifier
│
├── observers/               # 量化参数观测器
│   ├── base.py              # Observer 基类
│   ├── min_max.py           # MinMax
│   ├── mse.py               # MSE
│   ├── percentile.py        # Percentile
│   ├── kl.py                # KL 散度
│   └── lct.py               # Learned Clipping Threshold
│
├── modeling/                # 模型工具
│   ├── fuse.py              # norm 融合 / embedding 归零
│   ├── replace.py           # parametrization 折叠 / LayerNorm 替换
│   ├── moe_context.py       # MoE 校准上下文
│   ├── patch/               # 模型 forward patch
│   └── *_moe.py             # MoE 模型特化 (DeepSeek-V3, Qwen3, GLM4, ...)
│
├── datasets/                # 数据集加载
│   ├── __init__.py          # get_processed_dataset / get_calibration_dataloader
│   └── utils.py
│
├── train/                   # 训练支持
│   ├── fsdp_trainer.py      # FSDP Trainer
│   └── train_utils.py       # 训练工具
│
├── transformers/            # HuggingFace 集成
├── pytorch/                 # PyTorch 底层工具
└── utils/                   # 通用工具函数
```

---

## 6. 外部依赖关系

```
llm-compressor
  ├── compressed-tensors          # 量化/变换底层数据结构
  │   ├── TransformScheme/Args    #   旋转方案
  │   ├── QuantizationConfig      #   量化配置
  │   └── apply_transform_config  #   旋转应用
  ├── transformers (HuggingFace)  # 模型加载/保存
  ├── datasets (HuggingFace)      # 数据集加载
  ├── torch                       # PyTorch 基础
  ├── pydantic                    # Modifier 配置验证
  └── loguru                      # 日志
```

---

## 7. 待补充章节 (TODO)

以下章节将根据需要逐步展开：

- [ ] SequentialPipeline 详细实现（trace_subgraphs、子图切分、激活缓存管理）
- [ ] GPTQModifier 详细实现（Hessian 收集、量化算法、hooks 注册）
- [ ] QuantizationModifier 详细实现（config_groups、scheme 解析）
- [ ] Recipe 解析详细流程（YAML → Modifier 对象）
- [ ] ModelFree PTQ 流程
- [ ] 训练模式 (FSDP Trainer) 架构
- [ ] MoE 模型特化支持
- [ ] Observer 工作机制
- [ ] compressed-tensors 接口层

---

## 8. transform 相关 Modifier 设计与实现

### 8.1 目录结构与主类

```
src/llmcompressor/modifiers/transform/
├── __init__.py
├── spinquant/
│   ├── __init__.py
│   ├── base.py           # SpinQuantModifier 主类（R1-R4旋转/生命周期/注册/融合）
│   ├── mappings.py       # SpinQuantMapping 数据类+注册表
│   └── norm_mappings.py  # NormMapping 数据类+注册表
├── quip/
│   ├── __init__.py
│   └── base.py           # QuIPModifier 主类（全局旋转/生命周期）
├── smoothquant/
│   ├── __init__.py
│   ├── base.py           # SmoothQuantModifier 主类（激活平滑/生命周期/钩子）
│   ├── utils.py          # 各模型默认映射注册
│   └── README.md         # 映射配置说明
```

### 8.2 SpinQuantModifier（spinquant/base.py）

- **核心功能**：实现 SpinQuant/QuaRot/OSTQuant 四类旋转（R1-R4），支持 learnable/非学习两种模式，自动融合可折叠旋转，自动注册模型映射。
- **生命周期**：
  - `on_initialize`：推断映射，构建 R1-R4 TransformScheme，生成 TransformConfig。
  - `on_start`：解绑 embedding，均值归零（可选），fuse norm，应用旋转（可折叠的直接融合，在线的注册 forward hook）。
  - `on_end`：调用 `_fold_transforms_into_weights`，将可折叠旋转永久写入权重。
  - `on_finalize`：收尾，确保所有旋转已融合。
- **参数**：
  - `rotations`：启用哪些旋转（R1/R2/R3/R4）。
  - `transform_type`：旋转矩阵类型（hadamard/random-hadamard/random-matrix/identity）。
  - `learnable`：是否可学习（True=SpinQuant/OSTQuant，False=QuaRot）。
  - `transform_block_size_*`：各旋转的 block size。
  - `mappings`/`norm_mappings`：可自定义或自动推断。
- **注册机制**：
  - `SpinQuantMapping`/`NormMapping` 数据类，支持正则表达式，自动推断模型结构。
  - 支持用户自定义注册，未注册模型 fallback 到默认映射。
- **融合与钩子**：
  - 可折叠旋转（weight_input/weight_output）通过 torch parametrization 融合，最终调用 `replace_parametrizations_to_weights` 写回权重。
  - 在线旋转（input/output/q_attn/k_cache）注册 forward/pre-hook，推理时动态执行。

### 8.3 QuIPModifier（quip/base.py）

- **核心功能**：实现 QuIP/QuIP# 全局旋转，支持输入/输出两侧（v/u）旋转，支持 hadamard/random-hadamard/random-matrix。
- **生命周期**：
  - `on_initialize`：构建 TransformConfig（v/u 两组 scheme）。
  - `on_start`：自动解绑 embedding（如需），应用旋转。
  - `on_end`/`on_finalize`：收尾。
- **参数**：
  - `rotations`：["v", "u"]，分别控制输入/输出侧旋转。
  - `targets`：目标层（默认 "Linear"）。
  - 其余参数同 SpinQuant。
- **实现细节**：
  - v/u 分别对应 input/weight_input、weight_output/output 四种 TransformArgs。
  - 可折叠部分自动融合，在线部分注册 hook。

### 8.4 SmoothQuantModifier（smoothquant/base.py）

- **核心功能**：实现 SmoothQuant 激活平滑，自动推断/自定义映射，支持多模型结构，自动注册 forward hook 采集激活分布，校准后推入权重。
- **生命周期**：
  - `on_initialize`：推断/加载映射，注册采集钩子。
  - `on_start`：注册 forward hook。
  - `on_event`：校准阶段采集激活分布，结束后调用 `_apply_smoothing`，将平滑参数写入权重。
  - `on_end`/`on_finalize`：移除钩子，清理缓存。
- **参数**：
  - `smoothing_strength`：平滑强度（0-1）。
  - `mappings`：自定义映射（支持正则）。
  - `ignore`：忽略层。
- **注册机制**：
  - `utils.py` 维护各主流模型的默认映射注册表，自动推断。
  - 支持自定义映射，详见 smoothquant/README.md。
- **实现细节**：
  - forward hook 采集每层激活的 min/max，校准后计算平滑 scale，分别作用于激活和后续权重。
  - 支持 MoE/多专家等复杂结构。

### 8.5 统一设计要点

- **TransformScheme/TransformArgs**：所有 Modifier 统一用 compressed-tensors 的 TransformScheme/Args/Config 体系描述旋转/平滑方案，支持 block-wise、requires_grad、precision、targets、location、inverse 等参数。
- **自动映射与注册**：支持正则表达式，自动推断模型结构，极大提升适配性。
- **融合与钩子机制**：可折叠部分自动融合进权重，在线部分注册高效 hook，推理时零/极低开销。
- **生命周期钩子**：所有 Modifier 遵循统一的 on_initialize/on_start/on_event/on_end/on_finalize 生命周期，便于 pipeline 统一调度。
