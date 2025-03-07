
import pandas as pd

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


process_4_2_2()