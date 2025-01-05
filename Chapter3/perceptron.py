from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

## IRIS 데이터 셋 불러오기
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

## 데이터중 30%를 테스트 데이터 세트로, 70%를 훈련 데이터 세트로 설정
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

## 데이터 스케일링을 통해 표준화 진행
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

## Perceptron 훈련
ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)
y_pred = ppn.predict(X_test_std)
print('잘못 분류된 샘플 개수: %d' % (y_test != y_pred).sum())

print('정확도: %.3f' % accuracy_score(y_test, y_pred))

# ppn.score : predict, accuracy_score를 한번에 실행
# print('정확도: %.3f' % ppn.score(X_test_std, y_test))