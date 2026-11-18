# 复现与核验记录

2026 年 11 月 18 日，在 Windows、CPython 3.13.14 和 CPU 环境下，新建虚拟环境，按[依赖清单](../requirements.txt)从本地缓存的同版本 wheel 安装。`python -m pip check` 通过。参考版本包括 NumPy 2.5.3、pandas 3.1.1、scikit-learn 1.9.1、CatBoost 1.2.10 和 Matplotlib 3.11.2。

## 实际执行的检查

| 检查 | 方法与结果 |
|---|---|
| 离线输入 | 从工作目录的 `data/` 读取三份原始 CSV，重新核对换行标准化后的 SHA-256。训练集为 1,460 × 81，测试集为 1,459 × 80。 |
| 在线输入与失败路径 | 在没有本地数据、没有指定目录的独立工作目录执行 Notebook 的实际读取单元。固定版本归档下载成功，三份哈希与离线输入一致。故意改变一个字节后，校验函数抛出 `ValueError`。显式指定不存在的目录收到 `FileNotFoundError`。 |
| 主 Notebook | 在只备好 Notebook 与 `data/` 的目录执行 `jupyter execute --inplace --timeout=900 house_prices.ipynb`。12 个代码单元按顺序执行完毕，得到 5 份结果 CSV 和 6 张 PNG，Notebook 内也保存了六图。 |
| 重训结果与原版档案 | `fold_scores.csv`、`oof_predictions.csv`、`submission.csv` 与原版项目逐字节一致。在已有模型汇总表末尾增补两个 P90 列，原列按浮点容差复算一致。 |
| 折、标签与基线 | 仅凭保存的逐房表重建 `KFold(5, shuffle=True, random_state=42)` 的 5 组索引。与原始训练表按 Id 对齐，核对每个真实成交价及四项难例属性。用各训练折重算街区中位价，1,460 笔均与保存值相符。 |
| 指标与难例 | 独立从折外估价复算 15 个折分数、五折均值与样本标准差、全体 Log RMSE、MAE、百分比误差中位数及两种 P90。复算逐房估价方向和折内对数平方误差占比。 |
| 提交格式 | 回读主提交文件，验证列名是 `Id,SalePrice`，1,459 个 Id 按样表顺序排列，从 1461 到 2919，每个编号出现一次且预测价均为有限正数。使用另一个 CSV 读取器再次核对完整编号序列。 |
| 扩展实验 | 在同一新环境内重跑了扩展 Notebook 的 17 个代码单元和六张图。重建 12 个候选的 60 个折分数，按 1,460 条 OOF 对齐主实验，并复算 2010 年 175 条按年预测。此次扩展提交与原扩展提交逐字节一致。 |
| 文档及图表 | 检查 README、报告和 Notebook Markdown 单元的本地链接与图像路径，包含文件名大小写。校验目录清单、PNG 解码和图数，另检查了六张主图的实际渲染。 |
| Git 压缩包 | 用 `git archive` 导出审阅版，在独立空目录解压。包内有 27 个文件，逐一检查的 59 处相对链接均可解析。对解压件运行随仓库提供的脚本并对照原始数据，15 个折分数、1,460 笔折外记录和 1,459 笔提交预测均通过核验。CSV 保留原始字节，解压后的两份主实验预测文件与下表 SHA-256 一致。 |

主实验可独立运行[核验脚本](../scripts/verify_results.py)。第一条命令只依赖随仓库提供的表和图，第二条还检查原始标签、难例属性、各折街区基线与样表。

```bash
python scripts/verify_results.py
python scripts/verify_results.py --data-dir data
```

## 数据指纹

原始 CSV 将 `CRLF` 统一为 `LF` 后计算 SHA-256。主 Notebook 在读取时核对下表前三项，后两项记录本次生成结果的文件摘要。

| 文件 | SHA-256 |
|---|---|
| `train.csv` | `1e18addf81e5e4d347cc17ee6075bbe4a42b7fa26b9e5b063e8f692a5f929d41` |
| `test.csv` | `8fdd3d829d4d986b58f845c9553b225e67dd8383624d90fb6ca1d4bed5798c1e` |
| `sample_submission.csv` | `f19e99699b925aa3085c8d2ae43c62f15fbcd35e4a953b98025f8392c3ffa807` |
| 主实验 `submission.csv` | `866b6479d508f8fd43aad2ea852ded16a99fa549955f94e2e6e078fd29381104` |
| 主实验 `oof_predictions.csv` | `cd42a068aac0f60a9b4be052a6b594ab02503528988809f46187c5712524a144` |

## 与已有记录的核对

提供的三模型数值在所列精度下全部复现。`0.125347` 是主实验 CatBoost 的五折平均值，`0.126264` 是将它的全部折外预测汇总计算的 Log RMSE。`0.118185` 属于另一次已完成的三模型融合，详见[扩展实验](../extensions/ensemble/README.md)。第 3 折两个极端误差的 44.41% 只对应主实验 CatBoost。汇总表新增的 P90 以及[逐房误差表](../results/catboost_errors.csv)给出了核查入口。

本次报告记录了 Windows CPU 的实际执行。跨平台命令供复现使用，macOS 和 Linux 的独立重跑仍待验证。提交文件已完成本地检查。**尚未获得官方榜单成绩。**
