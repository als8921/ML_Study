
import pandas as pd
import numpy as np

df = pd.DataFrame([['green', 'M', 10.1, 'class2'],
                ['red', 'L', 13.5, 'class1'],
                ['blue', 'XL', 15.3, 'class2']])

df.columns = ['color', 'size', 'price', 'classlabel']
print(df.values)

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


def process_4_2_4():
    from sklearn.preprocessing import LabelEncoder
    
    X = df[['color', 'size', 'price']].values
    color_le = LabelEncoder()
    X[:, 0] = color_le.fit_transform(X[:, 0])   # 0열을 LabelEncoder를 사용하여 레이블링
    # print(X)

    #####################
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OrdinalEncoder

    ord_enc = OrdinalEncoder(dtype=int)
    col_trans = ColumnTransformer([('ord_enc', ord_enc, ['color'])])
    X_trans = col_trans.fit_transform(df)
    # print(X_trans)
    # print(col_trans.named_transformers_['ord_enc'].inverse_transform(X_trans))

    ###############
    # OneHotEncoder 를 사용한 특성의 고유값을 만들어 변환
    from sklearn.preprocessing import OneHotEncoder

    X = df[['color', 'size', 'price']].values
    color_ohe = OneHotEncoder()
    # print(color_ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray())

    ####################################################################################################
    # 
    # 1. ColumnTransformer 를 사용하여 blue, green, red [0, 0, 0] 행을 만들어 특성으로 넣기
    from sklearn.compose import ColumnTransformer

    # 4_2_2 map 함수를 사용하여 사이즈 데이터를 정수로 변환
    size_mapping = {'XL': 3, 'L': 2, 'M': 1}
    df['size'] = df['size'].map(size_mapping)

    X = df[['color', 'size', 'price']].values
    c_transf = ColumnTransformer([ ('onehot', OneHotEncoder(), [0]),
                                ('nothing', 'passthrough', [1, 2])])
    print(c_transf.fit_transform(X).astype(float))
    # OneHotEncoder에서 다중 공선성 문제 처리 (blue 열 삭제)
    c_transf = ColumnTransformer([ ('onehot', OneHotEncoder(categories='auto', drop='first'), [0]),
                                ('nothing', 'passthrough', [1, 2])])
    print(c_transf.fit_transform(X).astype(float))

    # 2. pd.get_dummies() 함수를 사용하여 위의 기능 구현
    print(pd.get_dummies(df[['price', 'color', 'size']], columns=['color']))
    print(pd.get_dummies(df[['price', 'color', 'size']], columns=['color'], drop_first=True))

    ###################################################################################################
    # 순서가 있는 특성 인코딩
    ddf = pd.DataFrame([['green', 'M', 10.1, 'class2'],
                   ['red', 'L', 13.5, 'class1'],
                   ['blue', 'XL', 15.3, 'class2']])

    ddf.columns = ['color', 'size', 'price', 'classlabel']
    print(ddf)
    ddf['x > M'] = ddf['size'].apply(lambda x: 1 if x in {'L', 'XL'} else 0)
    ddf['x > L'] = ddf['size'].apply(lambda x: 1 if x == 'XL' else 0)

    del ddf['size']
    print(ddf)
process_4_2_4()