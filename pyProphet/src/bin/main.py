'''
Created on 2018/09/17

@author: YOU_F
'''

import pandas as pd
import numpy as np
import datetime as dt
from pandas.core.frame import DataFrame
import matplotlib.pyplot as pyplot
import fbprophet as fbp

def main():
    ###log計算の実施フラグ
    log_flg = 1
    ###日単位変換の実施フラグ
    day_flg = 0
    ###インプットファイルパス
    input_file = "input/bitflyer_out.csv"

    ###CSVをDataframeに読み込む
    df = pd.read_csv(filepath_or_buffer=input_file, header=0, encoding="utf-8", sep=",", index_col=0, parse_dates=True)

    ###log変換用のDataframeをコピー
    if log_flg==1:
#         df_log = df.copy()
        df["open"] = np.log(df["open"])
        df["high"] = np.log(df["high"])
        df["low"] = np.log(df["low"])
        df["close"] = np.log(df["close"])
#         plot_df(df_log)

    ###Prophet用のDataframeを作成
    ###df：datetime
    ###y:対象データ
    df_p = pd.DataFrame({'ds' : df.index,
                         'y' : df["close"]})
    ###日単位データに変換
    if day_flg==1:
        df_p = df_p.resample('D', how='last')
        df_p = df_p.drop(columns='ds')
        df_p["ds"] = df_p.index

    print(df_p)

    ###モデルを作成(http://ill-identified.hatenablog.com/entry/2018/05/28/020224)
    model = fbp.Prophet(growth='linear',
                        changepoints=None,
                        n_changepoints=7,
                        yearly_seasonality='auto',
                        weekly_seasonality='auto',
                        daily_seasonality='auto',
                        holidays=None,
                        seasonality_prior_scale=10.0,
                        holidays_prior_scale=10.0,
                        changepoint_prior_scale=0.05,
                        mcmc_samples=0,
                        interval_width=0.80,
                        uncertainty_samples=1000)
    ###モデルにデータフレームを適用
    model.fit(df_p)
    future = model.make_future_dataframe(periods=60)
    forecast = model.predict(future)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
    model.plot(forecast).savefig('target/1.png')
    model.plot_components(forecast).savefig('target/2.png');


def date_parce(df = DataFrame):
    #t_date:tmpdate
    t_date = []
    for day in df["date"]:
        tmp = dt.datetime.strptime(day, '%Y-%m-%d %H:%M:%S')
        t_date.append(tmp)

    #t_dateをデータフレームに格納する
    df["date"] = t_date

    return df


def date_parce_float(df = DataFrame):
    tmp = df["date"].values.astype("datetime64[D]")
    #tmpの値をfloat型で格納する
#     df["date"] = tmp.astype(float)
    df["date"] = tmp

    return df


def plot_df(df = DataFrame):
    #値のプロット
    df.plot()
    #グラフの凡例
    pyplot.legend()
    #グラフの表示
    pyplot.show()



if __name__ == '__main__':
    main()