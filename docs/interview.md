# Evidence-backed portfolio use

**Resume bullets**

- Built a three-layer Triton reduction qualification harness separating CPU
  interpreter semantics, 24 explicit-target compiler configurations, and real
  RTX 4090 execution across 140 numerical/layout cases with repeated checks.
- Reduced invalid reduction-axis behavior to a standalone TTIR verifier fixture;
  retained legal boundaries and TTIR/TTGIR/LLVM/PTX/cubin artifacts without GPU
  driver discovery in offline compilation.
- Validated GPU memory/synchronization with NVIDIA memcheck, racecheck and synccheck;
  compared warp layouts using exact cubin hashes, SASS/resource records and a
  paired `torch.sum` baseline, retaining the no-benefit eight-warp result.

**面试问题**

- 解释器为什么不验证编译器？它绕过 lowering、layout、LLVM 和目标指令生成。
- 为什么负例不是“发现了编译器 bug”？输入违反轴范围规则，正确行为就是稳定诊断；项目验证这一边界。
- 为什么 GPU 隐藏后仍能编译？显式提供 target，并走不需要驱动的 AST/IR 编译入口；报告还检查驱动未初始化。
- 为什么 masked load 用 0？加法归约的单位元是 0，padding 不应改变和。
- 为什么 8 warps 寄存器更少但不更快？线程数量、shared memory、调度、带宽和 launch 下限共同影响耗时。
- 为什么不能把 graph timing 写成 eager API 延迟？它排除了 Python 调用和捕获阶段。
- 为什么 RTX 4090 不能代替 H100 验证？sm_89 和 sm_90 的指令与资源约束不同。

These are local qualification results, not an upstream accepted patch or a
production GPU compiler backend implementation.
