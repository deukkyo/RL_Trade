import os
import pandas as pd
import numpy as np

from proj import settings


COLUMNS_CHART_DATA = ['date', 'open', 'high', 'low', 'close', 'volume']

COLUMNS_TRAINING_DATA_V1 = [
    'open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
    'close_lastclose_ratio', 'volume_lastvolume_ratio',
    'close_ma5_ratio', 'volume_ma5_ratio',
    'close_ma10_ratio', 'volume_ma10_ratio',
    'close_ma20_ratio', 'volume_ma20_ratio',
    'close_ma60_ratio', 'volume_ma60_ratio',
    'close_ma120_ratio', 'volume_ma120_ratio',
]


def preprocess(data):
    windows = [5, 10, 20, 60, 120]
    for window in windows:
        data[f'close_ma{window}'] = data['close'].rolling(window).mean()
        data[f'volume_ma{window}'] = data['volume'].rolling(window).mean()
        data[f'close_ma{window}_ratio'] = \
            (data['close'] - data[f'close_ma{window}']) / data[f'close_ma{window}']
        data[f'volume_ma{window}_ratio'] = \
            (data['volume'] - data[f'volume_ma{window}']) / data[f'volume_ma{window}']
        
    data['open_lastclose_ratio'] = np.zeros(len(data))
    data.loc[1:, 'open_lastclose_ratio'] = \
        (data['open'][1:].values - data['close'][:-1].values) / data['close'][:-1].values
    data['high_close_ratio'] = (data['high'].values - data['close'].values) / data['close'].values
    data['low_close_ratio'] = (data['low'].values - data['close'].values) / data['close'].values
    data['close_lastclose_ratio'] = np.zeros(len(data))
    data.loc[1:, 'close_lastclose_ratio'] = \
        (data['close'][1:].values - data['close'][:-1].values) / data['close'][:-1].values
    data['volume_lastvolume_ratio'] = np.zeros(len(data))
    data.loc[1:, 'volume_lastvolume_ratio'] = (
        (data['volume'][1:].values - data['volume'][:-1].values) 
        / data['volume'][:-1].replace(to_replace=0, method='ffill')\
            .replace(to_replace=0, method='bfill').values
    )

    return data


def load_data(date_from, date_to):

    '''   csv 파일을 읽어 DataFrame 형식으로 저장
    df = pd.read_csv(
        os.path.join(settings.BASE_DIR, 'data', ver, f'{code}.csv'),
        thousands=',', header=header, converters={'date': lambda x: str(x)})
    '''

    df = pd.read_excel("C:\\Users\\Knayeon\\IdeaProjects\\Futures_Trade\\data")

    df.columns = ['date','time', 'open', 'high', 'low', 'close', 'volume']


'''
    # 데이터 전처리
    df = preprocess(df)
    
    # 기간 필터링
    df['date'] = df['date'].str.replace('-', '')
    df = df[(df['date'] >= date_from) & (df['date'] <= date_to)]
    df = df.fillna(method='ffill').reset_index(drop=True)

    # 차트 데이터 분리
    chart_data = df[COLUMNS_CHART_DATA]

    # 학습 데이터 분리
    training_data = None
    if ver == 'v1':
        training_data = df[COLUMNS_TRAINING_DATA_V1]
    elif ver == 'v1.1':
        training_data = df[COLUMNS_TRAINING_DATA_V1_1]
    elif ver == 'v2':
        df.loc[:, ['per', 'pbr', 'roe']] = df[['per', 'pbr', 'roe']].apply(lambda x: x / 100)
        training_data = df[COLUMNS_TRAINING_DATA_V2]
        training_data = training_data.apply(np.tanh)
    else:
        raise Exception('Invalid version.')
'''
    return chart_data, training_data


