"""从保存的逐房预测独立复算主实验，可选用原始竞赛 CSV 对照标签和样表。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

MODELS = ['Neighborhood median', 'Ridge', 'CatBoost']
ROOT = Path(__file__).resolve().parents[1]
FIGURES = ['01_sale_price.png', '02_missingness.png', '03_area_and_price.png',
           '04_model_comparison.png', '05_predicted_vs_actual.png', '06_error_by_price_range.png']


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(actual, expected, label: str) -> None:
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-8, err_msg=label)


def verify(results: Path, data: Path | None) -> None:
    oof = pd.read_csv(results / 'oof_predictions.csv')
    folds = pd.read_csv(results / 'fold_scores.csv').set_index(['Fold', 'Model'])
    summary = pd.read_csv(results / 'model_comparison.csv').set_index('Model')
    audit = pd.read_csv(results / 'catboost_errors.csv').set_index('Id')
    require(all(table.index.is_unique for table in (folds, summary, audit)), '折、模型和误差表的主键须唯一')
    submission = pd.read_csv(results / 'submission.csv')
    require(list(oof) == ['Id', 'SalePrice', 'Fold', *MODELS], '逐房表列不匹配')
    require(len(oof) == 1460 and oof.Id.nunique() == len(oof), '训练编号或行数不匹配')
    require(set(summary.index) == set(MODELS) and len(folds) == 15, '三个模型应各有五个折分数')
    numbers = oof[['SalePrice', *MODELS]].to_numpy()
    require(np.isfinite(numbers).all() and (numbers > 0).all(), '价格须为正的有限数值')
    expected_splits = list(KFold(n_splits=5, shuffle=True, random_state=42).split(oof))
    for index, (learn, valid) in enumerate(expected_splits, 1):
        require(len(learn) == 1168 and len(valid) == 292, '训练或验证折行数错误')
        np.testing.assert_array_equal(np.flatnonzero(oof.Fold.eq(index)), valid)

    y = oof.SalePrice.to_numpy()
    for model in MODELS:
        p = oof[model].to_numpy()
        squared = np.square(np.log(p) - np.log(y))
        usd = np.abs(p - y)
        percent = 100 * usd / y
        scores = []
        for index, (_, valid) in enumerate(expected_splits, 1):
            rmse = np.sqrt(squared[valid].mean())
            record = folds.loc[(index, model), :]
            require(record.ValidationRows == len(valid), f'{model} 第 {index} 折行数错误')
            close(record.LogRMSE, rmse, f'{model} 第 {index} 折 RMSE')
            scores.append(rmse)
        statistics = {'MeanLogRMSE': np.mean(scores), 'FoldSD': np.std(scores, ddof=1),
                      'OOFLogRMSE': np.sqrt(squared.mean()), 'MAE_USD': usd.mean(),
                      'MdAPE_pct': np.median(percent), 'P90_APE_pct': np.quantile(percent, .90),
                      'P90_AbsUSD': np.quantile(usd, .90)}
        for name, value in statistics.items():
            close(summary.loc[model, name], value, f'{model} 的 {name}')

    require(set(audit.index) == set(oof.Id), '误差明细的 Id 与逐房预测不一致')
    audit = audit.loc[oof.Id]
    np.testing.assert_array_equal(audit.Fold, oof.Fold)
    close(audit.SalePrice, y, '误差明细成交价')
    close(audit.CatBoost, oof.CatBoost, '误差明细预测价')
    signed = oof.CatBoost.to_numpy() - y
    pct = 100 * np.abs(signed) / y
    log_sse = (np.log(oof.CatBoost) - np.log(y)) ** 2
    close(audit.SignedError_USD, signed, '误差方向')
    close(audit.AbsError_USD, np.abs(signed), '绝对美元误差')
    close(audit.APE_pct, pct, '绝对百分比误差')
    close(audit.SquaredLogError, log_sse, '对数平方误差')
    close(audit.ShareOfFoldLogSSE_pct, 100 * log_sse / log_sse.groupby(oof.Fold).transform('sum'), '折内贡献')
    np.testing.assert_array_equal(audit.AboveP90_APE, pct > np.quantile(pct, .90))
    require(list(submission) == ['Id', 'SalePrice'] and len(submission) == 1459, '提交列名或行数错误')
    np.testing.assert_array_equal(submission.Id, np.arange(1461, 2920))
    require(submission.Id.is_unique and np.isfinite(submission.SalePrice).all() and submission.SalePrice.gt(0).all(), '提交值无效')
    for name in FIGURES:
        with (results / 'figures' / name).open('rb') as image:
            require(image.read(8) == b'\x89PNG\r\n\x1a\n', f'PNG 无效：{name}')

    if data is not None:
        train = pd.read_csv(data / 'train.csv')
        test = pd.read_csv(data / 'test.csv')
        sample = pd.read_csv(data / 'sample_submission.csv')
        pd.testing.assert_frame_equal(oof[['Id', 'SalePrice']], train[['Id', 'SalePrice']])
        pd.testing.assert_series_equal(submission.Id, test.Id)
        pd.testing.assert_series_equal(submission.Id, sample.Id)
        require(list(submission) == list(sample), '样表列名不一致')
        fields = ['Neighborhood', 'GrLivArea', 'OverallQual', 'SaleCondition']
        pd.testing.assert_frame_equal(audit[fields], train.set_index('Id')[fields])
        for learn, valid in expected_splits:
            by_area = train.iloc[learn].groupby('Neighborhood').SalePrice.median()
            expected = train.iloc[valid].Neighborhood.map(by_area).fillna(train.iloc[learn].SalePrice.median())
            close(oof.iloc[valid][MODELS[0]], expected, '训练折街区中位价')
        print('原始标签、难例属性、五折街区基线与提交样表均吻合。')
    pair_share = audit.loc[[524, 1299], 'ShareOfFoldLogSSE_pct'].sum()
    require(audit.loc[[524, 1299], 'Fold'].eq(3).all(), '代表房屋的折号有误')
    print(f'通过：3 个模型、15 个折分数、{len(oof):,} 条折外记录、{len(submission):,} 条提交预测。')
    print(f'CatBoost 平均 Log RMSE {summary.loc["CatBoost", "MeanLogRMSE"]:.6f}，'
          f'难例两笔占第 3 折对数平方误差的 {pair_share:.2f}%。')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT / 'results', help='主实验输出目录')
    parser.add_argument('--data-dir', type=Path, help='含原始 train、test 和 sample_submission 的目录')
    args = parser.parse_args()
    verify(args.results, args.data_dir)
