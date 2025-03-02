from models.Perceptron import Perceptron
import data.iris_data as iris_data
from models.AdalineGD import AdalineGD
from models.AdalineSGD import AdalineSGD


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
""" Chapter 2
- 인공 뉴런: 초기 머신 러닝의 간단한 역사
    - 인공 뉴런의 수학적 정의
    - 퍼셉트론 학습 규칙
- 파이썬으로 퍼셉트론 학습 알고리즘 구현
    - 객체 지향 퍼셉트론 API
    - 붓꽃 데이터셋에서 퍼셉트론 훈련
- 적응형 선형 뉴런과 학습의 수렴
    - 경사 하강법으로 비용 함수 최소화
    - 파이썬으로 아달린 구현
    - 특성 스케일을 조정하여 경사 하강법 결과 향상
    - 대규모 머신 러닝과 확률적 경사 하강법
- 요약
"""
### DATA ###
df = iris_data.df

y = df.iloc[0:100, 4].values # 꽃의 종류를 담은 배열
y = np.where(y == 'Iris-setosa', 0, 1)  # 값이 Iris-setosa 일 때 1로 아니면 0으로 만든 배열


# 꽃받침 길이와 꽃잎 길이를 추출합니다
X = df.iloc[0:100, [0, 2]].values   # iloc 으로 0번째 2번째 데이터를 담은 새로운 배열 제작
#############

def plot_decision_regions(X, y, classifier, resolution=0.02):

    # 마커와 컬러맵을 설정합니다
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # 샘플의 산점도를 그립니다
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolor='black')
        
def perceptron_train():


    ppn = Perceptron(eta=0.1, n_iter=10)

    ppn.fit(X, y)

    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of updates')

    # plt.savefig('images/02_07.png', dpi=300)
    plt.show()

    plot_decision_regions(X, y, classifier=ppn)
    plt.xlabel('Sepal length [cm]')
    plt.ylabel('Petal length [cm]')
    plt.legend(loc='upper left')


    #plt.savefig('images/02_08.png', dpi=300)
    plt.show()

def adalineGD_train():
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    ada1 = AdalineGD(n_iter=15, eta=0.1).fit(X, y)
    ax[0].plot(range(1, len(ada1.losses_) + 1), np.log10(ada1.losses_), marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(Mean squared error)')
    ax[0].set_title('Adaline - Learning rate 0.1')

    ada2 = AdalineGD(n_iter=15, eta=0.0001).fit(X, y)
    ax[1].plot(range(1, len(ada2.losses_) + 1), ada2.losses_, marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Mean squared error')
    ax[1].set_title('Adaline - Learning rate 0.0001')

    # plt.savefig('images/02_11.png', dpi=300)
    plt.show()

def adalineGD_std_train(): # 데이터 표준화를 통한 경사 하강법 성능 향상

    ################# 스케일 조정으로 인한 경사 하강법 결과 향상 #################
    # 특성을 표준화합니다.
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

    ada_gd = AdalineGD(n_iter=20, eta=0.5)
    ada_gd.fit(X_std, y)

    plot_decision_regions(X_std, y, classifier=ada_gd)
    plt.title('Adaline - Gradient descent')
    plt.xlabel('Sepal length [standardized]')
    plt.ylabel('Petal length [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    #plt.savefig('images/02_14_1.png', dpi=300)
    plt.show()

    plt.plot(range(1, len(ada_gd.losses_) + 1), ada_gd.losses_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Mean squared error')

    plt.tight_layout()
    #plt.savefig('images/02_14_2.png', dpi=300)
    plt.show()

def adalineSGD_train():
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

    ada_sgd = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
    ada_sgd.fit(X_std, y)

    plot_decision_regions(X_std, y, classifier=ada_sgd)
    plt.title('Adaline - Stochastic gradient descent')
    plt.xlabel('Sepal length [standardized]')
    plt.ylabel('Petal length [standardized]')
    plt.legend(loc='upper left')

    plt.tight_layout()
    #plt.savefig('figures/02_15_1.png', dpi=300)
    plt.show()

    plt.plot(range(1, len(ada_sgd.losses_) + 1), ada_sgd.losses_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Average loss')

    #plt.savefig('figures/02_15_2.png', dpi=300)
    plt.show()




adalineSGD_train()