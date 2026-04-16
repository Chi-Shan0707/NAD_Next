# 代码推理 vs 自然语言 CoT Pilot（2026-04-16）

英文版见：`docs/CODE_REASONING_COT_PILOT_20260416.md`

## 摘要

这轮工作把一个模糊直觉落成了一个可运行的 pilot：

- **核心假说**：在代码执行推理任务中，free-form natural-language CoT 可能不是理想的推理载体，因为它会把执行语义压平成线性叙事。
- **备选表示**：structured CoT、state table、execution trace、invariant / boundary-first reasoning 可能更稳，因为它们把状态与控制流外显化了。
- **本轮交付范围**：研究 framing、文献图谱、可证伪假说、pilot benchmark 生成器、prompt ablation、evaluation 设计，以及样例任务。

本项目的目标是 **code reasoning / execution**，不是 code generation。

## 交付物

- 根目录研究文档：
  - `research_summary.md`
  - `literature_map.md`
  - `hypotheses_and_rqs.md`
  - `experiment_plan.md`
  - `prompt_ablations.md`
  - `failure_taxonomy.md`
- 可运行脚本：
  - `pilot_benchmark.py`
- 样例数据：
  - `sample_tasks.jsonl`
- 入口文档：
  - `README.md`
  - `docs/README.md`

## 这个 pilot benchmark 包含什么

一共实现了 7 个合成 Python 程序族：

1. `sequential_baseline`
2. `single_branch`
3. `simple_loop`
4. `loop_plus_branch`
5. `phase_switch_loop`
6. `nested_loop`
7. `function_scope`

每道任务都记录以下可控属性：

- `branch_depth`
- `loop_nesting`
- `live_var_count`
- `phase_switch_count`
- `boundary_case_count`
- `scope_depth`
- `invariant_needed`

每道任务还包含：

- 可执行 `code`
- `entry_call`
- `gold_output`
- `gold_trace`
- `boundary_points`
- `expected_failure_modes`

## Gold trace 设计

`pilot_benchmark.py` 会真实执行生成出的程序，并导出 gold trace。每个 trace 至少含有：

- `call`、`line`、`return` 事件
- `locals_before` 与 `locals_after`
- `branch_taken`
- `loop_iter`
- `phase`
- `boundary_tags`

当前已经支持的关键边界标签包括：

- `first_iteration`
- `last_iteration`
- `iteration_at_pivot`
- `phase_switch`
- `first_post_switch_iteration`
- `helper_call_entry`
- `helper_return`
- `caller_resume`

这些信息足以支撑后续评测：

- final answer accuracy
- trace factuality
- trace validity
- per-step state accuracy
- branch-choice accuracy
- boundary-case error rate

## 已完成的验证

下面这条命令已经实际跑通：

```bash
python3 pilot_benchmark.py --num-per-family 3 --seed 0 --out sample_tasks.jsonl --pretty-sample 5
```

得到的结果是：

- 生成了 `21` 条任务
- `7` 个 family 都齐全
- `3` 个 difficulty level 都齐全
- 抽查中 `gold_output` 与入口函数返回值一致
- loop iteration 序列一致
- `phase_switch` 与 `caller_resume` 等边界标签在预期位置出现

此外还运行了仓库的验证脚本：

```bash
bash cookbook/00_setup/verify.sh
```

当前环境说明：

- benchmark 生成器本身可用
- 但本地环境缺少一些仓库通用依赖：`pyroaring`, `flask`, `plotly`, `hmmlearn`, `tokenizers`, `transformers`
- 因此 `verify.sh` 会报环境检查失败；这与本次 pilot 生成器本身无关

## 我这轮的主要发现

### 1. 这个想法是能被“研究化”的

最初的想法是一个直觉：  
“代码执行推理需要维护状态与控制流，而 free-form CoT 会把它们拍平成 prose。”

在整理文献之后，这个直觉已经可以被拆成几个明确、可检验的问题：

- 语义负载升高时，不同表示的退化速度是否不同？
- 错误是否主要集中在边界点与 phase switch？
- structured/stateful 表示是否主要改善 trace，而不只是 final answer？

也就是说，这已经不是哲学讨论，而是可以做成 controlled experiment 的问题。

### 2. 文献支持“显式状态有帮助”，但不直接证明“free-form CoT 不合适”

这是我认为最重要的判断。

目前文献更强地支持下面这件事：

- **program tracing 依赖显式状态维护**
- **working memory 是真实瓶颈**
- **外显化 tracing strategy / memory table 会帮助人类 tracing**
- **execution-aware 或 structured intermediate representation 对 LLM code reasoning 有帮助**

但它们还没有直接证明：

- “free-form natural-language CoT 本质上不适合代码执行推理”

所以更准确的说法是：

- **“显式状态 / trace 表示更好”比较有根据**
- **“free-form CoT 存在 representation mismatch”仍然需要实验来证伪或证成**

### 3. 最关键的实验切口不是 final answer，而是 trace quality

如果只看 final answer accuracy，这个方向很容易被误判。

因为模型可能：

- 最终答案答对，但 reasoning trace 是错的
- 最终答案答错，但 trace 里只在一个 boundary point 上失误

所以这个项目最有价值的地方是：  
它把评测从“只看答案”扩成了：

- trace factuality
- trace validity
- per-step state accuracy
- branch-choice accuracy
- boundary-case error rate
- failure taxonomy

我认为这是这轮工作最实质性的进展。

### 4. `phase_switch_loop` 和 `function_scope` 是最有信息量的任务族

从设计角度看，这两个 family 最能打到原始假说：

- `phase_switch_loop`：逼模型显式维护“阶段变化 + 迭代位置 + 切换边界”
- `function_scope`：逼模型显式维护“局部变量 / 调用边界 / 返回到 caller 后的恢复”

这两类任务比普通 sequential 或单层 loop 更有可能暴露 representation mismatch。

### 5. 这个方向更像一个 LLM evaluation 问题

它当然受 cognitive science 启发，也对 software-engineering explanation interface 有意义。  
但如果只看“这轮工作最接近什么研究问题”，我的判断是：

- **第一属性**：LLM evaluation / reasoning representation
- **第二属性**：cognitive-science inspired design
- **第三属性**：software-engineering-facing explanation design

也就是说，它的核心贡献更可能是：

- 提出一种更好的 code reasoning evaluation protocol
- 而不是直接提出新的 cognition theory

## 还没有被证明的部分

以下结论现在还不能说已经成立：

- free-form NL CoT 一定比 structured/stateful 表示差
- phase-switch 一定会显著放大 free-form CoT 的错误
- 所有 observed gain 都来自 representation，而不是 output-format regularization

这些都需要真正跑 prompt ablation 之后才能下结论。

## 我建议的下一步

最高优先级的下一步是补一个 **model-output scorer**，直接接在现有 pilot 后面：

- 解析 A/B/C/D prompt 输出
- 对齐到 `gold_trace`
- 计算各项指标
- 输出 family-level 与 failure-type-level 报告

具体优先级建议：

### Priority 1

- 同一模型、同一任务、同一批输入
- 跑 `A` free-form CoT / `B` structured CoT / `C` state table / `D` execution trace

### Priority 2

- 做 token-budget matched control
- 排除“只是因为某种格式更长、更规整”的解释

### Priority 3

- 把 pilot 发现迁移到更真实的外部 benchmark 子集
- 例如 CRUXEval 风格的 execution tasks

## 一句话总结

这轮工作最大的成果不是“已经证明了 free-form CoT 不行”，而是：

> 我们已经把这个想法变成了一个结构清楚、可复现、可运行、可被反驳的研究包。

这是从直觉走向研究最关键的一步。
