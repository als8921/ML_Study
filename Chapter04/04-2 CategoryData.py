
import pandas as pd
import numpy as np

df = pd.DataFrame([['green', 'M', 10.1, 'class2'],
                ['red', 'L', 13.5, 'class1'],
                ['blue', 'XL', 15.3, 'class2']])

df.columns = ['color', 'size', 'price', 'classlabel']


def process_4_2_1():
    pass

def process_4_2_2():
    # 사이즈를 정수 데이터로 변환하는 딕셔너리
    size_mapping = {'XL': 3, 'L': 2, 'M': 1}
    # map 함수를 사용하여 사이즈 데이터를 정수로 변환
    df['size'] = df['size'].map(size_mapping)
    print(df)

    # 정수를 사이즈 데이터로 변환하는 딕셔너리
    inv_size_mapping = {v: k for k, v in size_mapping.items()}
    # map 함수를 사용하여 정수 데이터를 사이즈 데이터로 변환
    df['size'] = df['size'].map(inv_size_mapping)
    print(df)


def process_4_2_3():
    # 클래스 레이블을 문자열에서 정수로 바꾸기 위해
    # 매핑 딕셔너리를 만듭니다
    class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))} # np.unique : set와 같은 기능 (중복 데이터 제거)

    # 클래스 레이블을 문자열에서 정수로 바꿉니다
    df['classlabel'] = df['classlabel'].map(class_mapping)
    print(df)

    # 클래스 레이블을 거꾸로 매핑합니다
    inv_class_mapping = {v: k for k, v in class_mapping.items()}
    df['classlabel'] = df['classlabel'].map(inv_class_mapping)
    print(df)

    #######

    from sklearn.preprocessing import LabelEncoder

    # 사이킷런의 LabelEncoder을 사용한 레이블 인코딩
    class_le = LabelEncoder()
    y = class_le.fit_transform(df['classlabel'].values)
    # 거꾸로 매핑
    print(y)
    print(class_le.inverse_transform(y))


process_4_2_3()