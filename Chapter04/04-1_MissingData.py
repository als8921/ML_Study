def process_4_1_1():
    import pandas as pd
    from io import StringIO

    csv_data = \
    '''A,B,C,D
    1.0,2.0,3.0,4.0
    5.0,6.0,,8.0
    10.0,11.0,12.0,'''

    df = pd.read_csv(StringIO(csv_data))
    """
        DF
        ------------------------
        A     B     C    D
        0   1.0   2.0   3.0  4.0
        1   5.0   6.0   NaN  8.0
        2  10.0  11.0  12.0  NaN
    """

    print(df.isnull().sum())
    # `values` 속성으로 넘파이 배열을 얻을 수 있습니다
    df.values

process_4_1_1()