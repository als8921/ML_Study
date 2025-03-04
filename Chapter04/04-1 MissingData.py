import pandas as pd
from io import StringIO

csv_data = \
'''A,B,C,D
1.0,2.0,3.0,4.0
5.0,,,8.0
10.0,11.0,12.0,'''

df = pd.read_csv(StringIO(csv_data))

def process_4_1_1():
    """
    데이터 값 확인
        df.isnull()       : 데이터 값이 null인지 True, False로 반환
        df.isnull().sum() : True 인 개수를 반환
        df.values         : 판다스 데이터프레임을 넘파이 배열로 전환
    """
    print(df.isnull())
    print(df.isnull().sum())
    print(df.values)

def process_4_1_2():
    """
    누락된 값 제거
        df.dropna(axis=0)     : null 데이터가 있는 행을 삭제
        df.dropna(axis=1)     : null 데이터가 있는 열을 삭제
        df.dropna(how='all')  : 행의 모든 값이 NaN 인 행을 삭제
        df.dropna(thresh=4)   : 행의 NaN의 개수가 thresh 보다 큰 행을 삭제
        df.dropna(subset=['C'])   : 특정 열에 Nan이 있는 행을 삭제
    """
    print(df.dropna(axis=0))
    print(df.dropna(axis=1))
    print(df.dropna(how='all'))
    print(df.dropna(thresh=4))
    print(df.dropna(subset=['C']))

def process_4_1_3():
    # 행의 평균으로 누락된 값 대체하기
    from sklearn.impute import SimpleImputer
    import numpy as np

    imr = SimpleImputer(missing_values=np.nan, strategy='mean')
    imr = imr.fit(df.values)
    imputed_data = imr.transform(df.values)
    print(imputed_data)
process_4_1_3()