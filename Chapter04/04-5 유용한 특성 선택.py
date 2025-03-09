import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from SequentialBackwardSelection import SBS
# 데이터 불러오기
##########################################################################################
df_wine = pd.read_csv('https://archive.ics.uci.edu/'
                      'ml/machine-learning-databases/wine/wine.data',
                      header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']

# X 에 특성 데이터 y 에 정답 클래스 데이터
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

# 테스터 데이터의 비율 0.3으로 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)
print("데이터 불러오기")


# L1 Regularization
##########################################################################################
# lr = LogisticRegression(penalty='l1', C=1.0, solver='liblinear')
# # C=1.0이 기본입니다.
# # 규제 효과를 높이거나 낮추려면 C 값을 증가시키거나 감소시킵니다.
# lr.fit(X_train_std, y_train)
# print('훈련 정확도:', lr.score(X_train_std, y_train))
# print('테스트 정확도:', lr.score(X_test_std, y_test))
# print(lr.intercept_)
# print(lr.coef_)

# L1 Regularization
##########################################################################################
# fig = plt.figure()
# ax = plt.subplot(111)

# colors = ['blue', 'green', 'red', 'cyan',
#           'magenta', 'yellow', 'black',
#           'pink', 'lightgreen', 'lightblue',
#           'gray', 'indigo', 'orange']

# weights, params = [], []
# for c in np.arange(-4., 6.):
#     lr = LogisticRegression(penalty='l1', C=10.**c, solver='liblinear',
#                             multi_class='ovr', random_state=0)
#     lr.fit(X_train_std, y_train)
#     weights.append(lr.coef_[1])
#     params.append(10**c)

# weights = np.array(weights)

# for column, color in zip(range(weights.shape[1]), colors):
#     plt.plot(params, weights[:, column],
#              label=df_wine.columns[column + 1],
#              color=color)
# plt.axhline(0, color='black', linestyle='--', linewidth=3)
# plt.xlim([10**(-5), 10**5])
# plt.ylabel('Weight coefficient')
# plt.xlabel('C (inverse regularization strength)')
# plt.xscale('log')
# plt.legend(loc='upper left')
# ax.legend(loc='upper center',
#           bbox_to_anchor=(1.38, 1.03),
#           ncol=1, fancybox=True)

# plt.show()

# SBS 알고리즘을 사용한 특성 선택
##########################################################################################

knn = KNeighborsClassifier(n_neighbors=5)

# 특성을 선택합니다
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)

# 선택한 특성의 성능을 출력합니다
k_feat = [len(k) for k in sbs.subsets_]

print(sbs.subsets_)
k3 = list(sbs.subsets_[10])

knn.fit(X_train_std, y_train)
print('훈련 정확도:', knn.score(X_train_std, y_train))
print('테스트 정확도:', knn.score(X_test_std, y_test))

knn.fit(X_train_std[:, k3], y_train)
print('훈련 정확도:', knn.score(X_train_std[:, k3], y_train))
print('테스트 정확도:', knn.score(X_test_std[:, k3], y_test))

plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
plt.show()