import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models.LogisticRegressionGD import LogisticRegressionGD

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # 마커와 컬러맵을 설정합니다.
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 결정 경계를 그립니다.
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # 클래스 샘플을 그립니다
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolor='black')

    # 테스트 샘플을 부각하여 그립니다.
    if test_idx:
        # 모든 샘플을 그립니다.
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='none',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100,
                    label='Test set')
        
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

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

X_train_01_subset = X_train_std[(y_train == 0) | (y_train == 1)]
y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]


def LogisticRegresssion():
    lrgd = LogisticRegressionGD(eta=0.3, n_iter=1000, random_state=1)
    lrgd.fit(X_train_01_subset,
            y_train_01_subset)

    plot_decision_regions(X=X_train_01_subset,
                        y=y_train_01_subset,
                        classifier=lrgd)

    plt.xlabel('Petal length [standardized]')
    plt.ylabel('Petal width [standardized]')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

def SKLearnLogisticRegression():
    ## sklearn의 Logistic Regressgion을 사용한 모델 훈련
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(C=100.0, solver='lbfgs', multi_class='ovr')
    # C : 규제 하이퍼파라미터의 역수로 C 값을 감소시킬 수록 규제 강도가 증가하여 Overfit을 방지할 수 있음
    # solver : 볼록 최적화 (convex optimization) 을 위한 알고리즘 'newton-cg, lbfgs, liblinear, sag, saga' 의 알고리즘이 있음
    lr.fit(X_train_std, y_train)

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    plot_decision_regions(X_combined_std, y_combined,
                        classifier=lr, test_idx=range(105, 150))
    plt.xlabel('Petal length [standardized]')
    plt.ylabel('Petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

def SKLearnSupportVectorMachine():
    from sklearn.svm import SVC

    svm = SVC(kernel='linear', C=1.0, random_state=1)
    svm.fit(X_train_std, y_train)

    plot_decision_regions(X_combined_std,
                        y_combined,
                        classifier=svm,
                        test_idx=range(105, 150))
    plt.xlabel('Petal length [standardized]')
    plt.ylabel('Petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

def SKLearnKernelSVM():
    from sklearn.svm import SVC

    svm = SVC(kernel='rbf', random_state=1, gamma=0.2, C=1.0)
    # Gamma 파라미터 값을 높일 수록 overfit 현상이 발생한다.
    svm.fit(X_train_std, y_train)

    plot_decision_regions(X_combined_std, y_combined,
                        classifier=svm, test_idx=range(105, 150))
    plt.xlabel('Petal length [standardized]')
    plt.ylabel('Petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    #plt.savefig('figures/03_15.png', dpi=300)
    plt.show()

def DecisionTree():

    from sklearn.tree import DecisionTreeClassifier

    tree_model = DecisionTreeClassifier(criterion='gini',
                                        max_depth=4,
                                        random_state=1)
    tree_model.fit(X_train, y_train)

    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(X_combined, y_combined,
                        classifier=tree_model,
                        test_idx=range(105, 150))

    plt.xlabel('Petal length [cm]')
    plt.ylabel('Petal width [cm]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    #plt.savefig('figures/03_20.png', dpi=300)
    plt.show()
    
    from sklearn import tree

    feature_names = ['Sepal length', 'Sepal width',
                    'Petal length', 'Petal width']
    tree.plot_tree(tree_model,
                feature_names=feature_names,
                filled=True)

    #plt.savefig('figures/03_21_1.pdf')
    plt.show()
    
    
def RamdomForest():
    from sklearn.ensemble import RandomForestClassifier
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))

    forest = RandomForestClassifier(n_estimators=25,
                                    random_state=1,
                                    n_jobs=2)
    forest.fit(X_train, y_train)

    plot_decision_regions(X_combined, y_combined,
                        classifier=forest, test_idx=range(105, 150))

    plt.xlabel('Petal length [cm]')
    plt.ylabel('Petal width [cm]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    #plt.savefig('figures/03_2.png', dpi=300)
    plt.show()

RamdomForest()