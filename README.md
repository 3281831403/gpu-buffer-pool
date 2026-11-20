# gpu-buffer-pool

GPU 统一显存池的字节级分配器修复题。仓库开局是纯桩环境：8 个函数全部 `raise NotImplementedError`，15 个测试全部红。

## ① 题面

渲染器换成统一显存池后，跑久了碎片率飙到 40%、句柄还撞车：对齐向下取整，越界写坏邻居；首次适配而不是最佳适配，碎片率 40%；释放不合并，空闲块越来越碎；整理后句柄失效，外部引用悬空；碎片率按块数算，忽略大小；扩容按固定 1 MB，超大请求永远失败；句柄重复分配，两次请求返回同一个；越界句柄不校验，释放别人的块；批量失败只回滚部分，留下半分配；alignment=0 不校验，除零。pool/allocator.py 是核心，tests/test_allocator.py 15个红。要修的是：对齐向上取整并拒绝非 2 的幂、最佳适配、释放即合并、整理后句柄稳定、按字节加权计算碎片率、按请求大小扩容、句柄唯一、校验非法句柄、批量失败整体回滚、alignment=0 抛 PoolError。

## ② 运行方式

在仓库根目录执行：

```powershell
python -m pytest tests -q
```

初始状态的输出尾部必须恰好是：

```text
15 failed in <时间>s
```

即恰好 15 个 `failed`，且没有 collection error、import error 或 syntax error。环境仅使用标准库（`bisect`、`collections`）和 pytest，离线运行，不依赖随机数。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 图形渲染 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## ④ 测试约束

`tests/test_allocator.py` 不可修改：不得删改测试、增加 skip/xfail、放宽断言，也不得通过测试侧的特殊分支绕过断言。修复应全部落在 `pool/allocator.py`，并保持 `PoolError` 的异常契约以及字节、对齐、句柄唯一性的 docstring 约定。
