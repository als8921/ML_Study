from Perceptron import Perceptron
import DataSet


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ppn = Perceptron(eta=0.1, n_iter=10)



df = DataSet.df

y = df.iloc[0:100, 4].values # 꽃의 종류를 담은 배열
y = np.where(y == 'Iris-setosa', 0, 1)  # 값이 Iris-setosa 일 때 1로 아니면 0으로 만든 배열


# 꽃받침 길이와 꽃잎 길이를 추출합니다
X = df.iloc[0:100, [0, 2]].values   # iloc 으로 0번째 2번째 데이터를 담은 새로운 배열 제작
ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')

# plt.savefig('images/02_07.png', dpi=300)
plt.show()