# 后续实验：同一五折上的融合模型

本目录回答一个后续问题。在原版 CatBoost 的基础上，融合多种估价方法能改善多少？沿用主实验的 1,460 套房、样本顺序和五折编号，尝试了 12 个单模型与固定权重的融合方案。最佳平均分来自稳健样条 Ridge、ElasticNet 和改进版 CatBoost。三者在对数价格上各占三分之一。[扩展 Notebook](house_prices.ipynb) · [全部候选](model_comparison.csv) · [主实验](../../README.md)

| 统一口径 | 原版 CatBoost | 三模型融合 |
|---|---:|---:|
| 五折平均 Log RMSE | 0.125347 | 0.118185 |
| 折间样本标准差 | 0.016982 | 0.020584 |
| 折外 MAE（美元） | 14,859 | 13,533 |
| 绝对百分比误差 P90 | 16.61% | 15.36% |

平均 Log RMSE 降低 5.71%，新版赢了四折。第 3 折从 0.150087 升至 0.151577，Id 1299 的估价从 519,731 升至 665,850 美元，实际成交价为 160,000 美元。固定原版相对误差最大的 146 套，其中 111 套的误差缩小。[逐折成绩](fold_scores.csv) · [逐房折外估价](oof_predictions.csv)

**按年复核。** 固定五折选出的方案后，用 2006–2009 年的 1,285 套房训练，再预测 2010 年的 175 套。Log RMSE 从 0.118136 降至 0.107017。2010 年交易参加过前面的随机五折选型，解释这次检查时需考虑这种样本重叠。[按年成绩](validation_2010.csv) · [2010 年逐房估价](predictions_2010.csv)

## 复现与文件

先按[根目录说明](../../README.md)安装依赖。下面的 PowerShell 命令从仓库根目录启动，假定三个原始 CSV 已放入根目录的 `data/`。单独的输出目录使重跑结果易于与随附 CSV 对照。

```powershell
$env:PYTHONUTF8 = '1'
$env:HOUSE_PRICES_DATA_DIR = (Resolve-Path data).Path
$env:HOUSE_PRICES_OUTPUT_DIR = Join-Path (Resolve-Path extensions/ensemble).Path 'rebuilt'
.\.venv\Scripts\jupyter.exe execute --inplace --timeout=1800 extensions/ensemble/house_prices.ipynb
```

在 macOS 或 Linux 上激活同一环境后，可执行下列命令。

```bash
HOUSE_PRICES_DATA_DIR="$PWD/data" HOUSE_PRICES_OUTPUT_DIR="$PWD/extensions/ensemble/rebuilt" \
  jupyter execute --inplace --timeout=1800 extensions/ensemble/house_prices.ipynb
```

[Notebook](house_prices.ipynb) 内有模型实现、预处理、六张图与执行输出。[模型比较](model_comparison.csv)、[逐折成绩](fold_scores.csv)、[训练房屋折外估价](oof_predictions.csv)、[2010 年指标](validation_2010.csv)、[2010 年逐房估价](predictions_2010.csv)和[融合提交文件](submission.csv)是本目录附带的六份数据表。执行后还会生成完整难例表和独立图片。主实验的 CatBoost 提交文件位于[根目录的 results](../../results/submission.csv)。

扩展特征参考了 Jessi-Liu 的质量等级与质量、面积交互实现，模型组合参考了 apoorva2121 的公开案例。回归器自行在本项目各训练折拟合，融合使用固定等权。[特征工程源码](https://github.com/Jessi-Liu/house-prices-prediction/blob/7af2013f58ad8e59e4ea9772ffdcbc9691e7ba68/src/feature_engineering.py) · [模型融合案例](https://github.com/apoorva2121/ames-house-prices/blob/7fe1337e46d5578ced679668a3136ab73aeabd1e/README.md) · [样条回归示例](https://scikit-learn.org/stable/auto_examples/linear_model/plot_polynomial_interpolation.html)
