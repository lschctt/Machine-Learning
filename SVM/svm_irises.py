import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

data_set = pd.read_excel('../data/Irises classification.xlsx', sheet_name='train')
x = data_set.values[:, [0, 1, 2]]
y = data_set.values[:, [5]]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
y_train = y_train.astype('int')
y_test = y_test.astype('int')
x_train = x_train.astype('double')
x_test = x_test.astype('double')


# =============== Create model ===============
def classifier():
    clf = svm.SVC(C=0.5,  # penalty term(惩罚因子):C
                  kernel='rbf',  # kernel='rbf':Gauss_kernel
                  decision_function_shape='ovr')  # 决策函数
    return clf


# =============== Evaluation ===============
def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    print(tip, "Accuracy:", np.mean(acc))


def print_accuracy(clf, xx_train, yy_train, xx_test, yy_test):

    # score(x_train, y_train): represent the accuracy of prediction on x_train
    print('training prediction:%.3f' % (clf.score(xx_train, yy_train)))
    print('test data prediction:%.3f' % (clf.score(xx_test, yy_test)))

    # predict(self, X): classify samples from X
    show_accuracy(clf.predict(x_train), yy_train, 'training data')
    show_accuracy(clf.predict(x_test), yy_test, 'testing data')

    # decision_function(self, X): Evaluate the decision function for the samples in X
    # give back a numpy array, each element represents whether the samples in X at the right or left of the hyperplane
    # and how far it is from the hyperplane
    print('decision_function:\n', clf.decision_function(xx_train))


if __name__ == "__main__":
    # define SVM model
    model = classifier()
    # train SVM model
    model.fit(x_train, y_train.ravel())

    print_accuracy(model, x_train, y_train, x_test, y_test)

    # if only two kinds of labels, we can do a visualization
    iris_feature = 'sepal length', 'sepal width', 'petal length', 'petal width'
    if x.shape[1] == 2:
        x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
        x2_min, x2_max = x[:, 1].min(), x[:, 1].max()

        x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]
        grid_test = np.stack((x1.flat, x2.flat), axis=1)
        print('grid_test.shape', grid_test.shape)

        grid_hat = model.predict(grid_test)
        print('grid_hat.shape', grid_hat.shape)

        grid_hat = grid_hat.reshape(x1.shape)
        print('grid_hat.shape', grid_hat.shape)

        cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
        cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])

        plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)
        # train samples
        plt.scatter(x_train[:, 0], x_train[:, 1], c=np.squeeze(y_train), edgecolor='k', s=50, cmap=cm_dark)
        # test samples
        plt.scatter(x_test[:, 0], x_test[:, 1], c=np.squeeze(y_test), s=120, facecolor='none', zorder=10, cmap=cm_dark)
        plt.xlabel(iris_feature[0], fontsize=20)
        plt.ylabel(iris_feature[1], fontsize=20)
        plt.xlim(x1_min, x1_max)
        plt.ylim(x2_min, x2_max)
        plt.title('svm in iris data classification', fontsize=30)
        plt.grid()
        plt.show()

    elif x.shape[1] == 3:
        fig = plt.figure()
        cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_train[:, 0], x_train[:, 1], x_train[:, 2], s=100, c=y_train, marker='.', cmap=cm_dark)
        ax.set_xlabel(iris_feature[0], fontsize=10)
        ax.set_ylabel(iris_feature[1], fontsize=10)
        ax.set_zlabel(iris_feature[2], fontsize=10)
        plt.show()







