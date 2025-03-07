import pandas as pd
from io import StringIO

csv_data = \
'''A,B,C,D
1.0,2.0,3.0,4.0
5.0,,1.0,8.0
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
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import FunctionTransformer
    import numpy as np

    # NaN이 있는 열의 평균을 내서 NaN 값을 대체
    imr = SimpleImputer(missing_values=np.nan, strategy='mean')
    imr = imr.fit(df.values)
    imputed_data = imr.transform(df.values)
    
    print("Origin")
    print(df.values)
    print("Imputed_data")
    print(imputed_data)


    # 행을 기준으로 평균을 내기 위해서 입력 행렬을 Transform 하고 계산 후 다시 Transform 시킴
    ftr_imr = FunctionTransformer(lambda X: imr.fit_transform(X.T).T)
    imputed_data = ftr_imr.fit_transform(df.values)
    print("Imputed_data")
    print(imputed_data)

    # add_indicator=True : 빈 값이 있는 열을 값이 있으면 0, Nan 값이면 1로 반환한 열을 기존 행렬에 추가
    imr = SimpleImputer(add_indicator=True)
    imputed_data = imr.fit_transform(df.values)
    print("Imputed_data")
    print(imputed_data)
    # imr.indicator_.features_ : 빈 값이 있는 열의 위치 리스트
    print(imr.indicator_.features_)

    # imr.indicator_.features_ 열에서의 누락된 값의 위치를 나타내는 배열을 반환 (Boolean)
    imr.indicator_.fit_transform(df.values)
    
    # 원본 특성으로 변환
    imr.inverse_transform(imputed_data)

    ###############
    # 실험적인 기능 IterativeImputer
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer

    iimr = IterativeImputer()
    print(iimr.fit_transform(df.values))

    # K 최근접 이웃 방법을 사용하여 누락된 값 채우기
    from sklearn.impute import KNNImputer

    kimr = KNNImputer()
    print(kimr.fit_transform(df.values))

    ##############
    # pandas fillna() 함수를 사용하여 누락된 데이터 대체
    print("PANDAS")
    print(df)
    print("df.fillna(df.mean())")
    print(df.fillna(df.mean())) # 열의 평균을 사용

    print("df.fillna(method='bfill')")
    print(df.fillna(method='bfill')) # 해당 열의 다음 값으로 대체

    print("df.fillna(method='ffill')")
    print(df.fillna(method='ffill')) # 해당 열의 이전 값으로 대체
    
    print("df.fillna(method='ffill', axis=1)")
    print(df.fillna(method='ffill', axis=1)) # 해당 행의 이전 값으로 대체 (축을 가로로)


process_4_1_3()