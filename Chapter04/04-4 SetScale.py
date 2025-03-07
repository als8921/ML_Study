import pandas as pd
import numpy as np

df_wine = pd.read_csv('https://archive.ics.uci.edu/'
                      'ml/machine-learning-databases/wine/wine.data',
                      header=None)

# UCI 머신러닝 저장소의 Wine 데이터셋에 접근되지 않을 때
# 다음 코드의 주석을 제거하고 로컬 경로에서 데이터셋을 읽으세요:

# df_wine = pd.read_csv('wine.data', header=None)


df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']

# print('Class labels', np.unique(df_wine['Class label']))
# print(df_wine.head())

from sklearn.model_selection import train_test_split

# X 에 특성 데이터 y 에 정답 클래스 데이터
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

# 테스터 데이터의 비율 0.3으로 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
####################################################################################################################### 04-3 내용

# 최소-최대 스케일 변환
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)

# 표준화
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

# 중간 값을 빼고 25백분위수와 75백분위수의 차이로 나누기
from sklearn.preprocessing import RobustScaler
rbs = RobustScaler()
X_train_robust = rbs.fit_transform(X_train)
X_test_robust = rbs.transform(X_test)

# 최대 절댓갑승로 나누기
from sklearn.preprocessing import MaxAbsScaler
mas = MaxAbsScaler()
X_train_maxabs = mas.fit_transform(X_train)
X_test_maxabs = mas.transform(X_test)

#######################################################################################################################
ex = np.array([0, 1, 2, 3, 4, 5])

print('standardized:', (ex - ex.mean()) / ex.std())
print('normalized:', (ex - ex.min()) / (ex.max() - ex.min()))

from sklearn.preprocessing import scale, minmax_scale, robust_scale, maxabs_scale
print('StandardScaler:', scale(ex))
print('MinMaxScaler:', minmax_scale(ex))
print('RobustScaler:', robust_scale(ex))
print('MaxAbsScaler:', maxabs_scale(ex))

#######################################################################################################################
from scipy import sparse
X_train_sparse = sparse.csr_matrix(X_train) # X_train 데이터를 CSR(Compressed Sparse Row) 형식의 희소 행렬로 변환
X_train_maxabs = mas.fit_transform(X_train_sparse)
X_train_robust = rbs.transform(X_train_sparse)


#######################################################################################################################
from sklearn.preprocessing import Normalizer
# 데이터를 (root square sum 값)으로 나눔
nrm = Normalizer() # norm='l2'
X_train_l2 = nrm.fit_transform(X_train)
# 데이터를 (데이터의 절댓값의 합)으로 나눔
nrm = Normalizer(norm='l1')
X_train_l1 = nrm.fit_transform(X_train)