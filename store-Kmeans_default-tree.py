import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn.metrics
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from yellowbrick.regressor import ResidualsPlot

# question 4.1

def StoreAnalysis():

    # read, wrangle and normalize data
    store_data = pd.read_csv('Stores.csv')
    store_data.dropna(axis=0, inplace=True)
    store_names = store_data['Store']
    store_data.drop(['Store'], axis=1, inplace=True)
    normalize = Normalizer()
    normalize.fit(store_data)
    store_data_normalized = pd.DataFrame(normalize.transform(store_data), columns=store_data.columns)
    store_data['Stores'] = store_names
    print(store_data_normalized.head())

    # KMeans model,
    k_array = np.arange(1, 11)
    inertias = []
    for k in k_array:
        store_model = KMeans(n_clusters=k, random_state=2021)
        store_model.fit(store_data_normalized)
        inertias.append(store_model.inertia_)
    # Plot inertias vs k and choose optimal k
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(k_array, inertias, '-o')
    ax[0].set(xlabel='Number of Clusters, k', ylabel='Inertia', title='Inertia vs Num Clusters')
    ax[0].set_xticks(k_array)

    # Choose k = 3, and add clusters to DF
    store_model = KMeans(n_clusters=3, random_state=2021)
    store_model.fit(store_data_normalized)
    store_data['clusters'] = store_model.labels_

    # plot histogram of cluster number
    ax[1].hist(store_data['clusters'], bins=5, color='navy')
    ax[1].set(xticks=[0, 1, 2], xlabel='cluster', ylabel='count', title='Number of Stores in each Cluster')
    plt.show()


def DefaultDecisionTree():
    # read in data, wrangle
    default_data = pd.read_csv('ccDefaults.csv')
    pd.set_option('display.width', None)
    pd.set_option('display.max_columns', None)
    default_data.dropna(axis=0, inplace=True, how='any')

    default_data.drop(['ID'], axis=1, inplace=True)
    print(default_data.corr())
    # pay 1, pay 2, pay 3, pay 4 have highest correlation with dpnm
    X_default = default_data[['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4']]
    y = default_data[['dpnm']]

    # normalize X
    normalize = Normalizer()
    normalize.fit(X_default)
    X = pd.DataFrame(normalize.transform(X_default), columns=X_default.columns)
    # split, build model and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=2021)
    model_tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=2021)
    model_tree.fit(X_train, y_train)

    # get accuracy scores and print
    accuracy_train = model_tree.score(X_train, y_train)
    accuracy_test = model_tree.score(X_test, y_test)
    print("Accuracy for the training data:", accuracy_train)
    print("Accuracy for the testing data:", accuracy_test)

    # plot confusion matrix and tree
    y_pred = model_tree.predict(X_test)
    cf = metrics.confusion_matrix(y_test, y_pred)
    display_cf = metrics.ConfusionMatrixDisplay(cf, display_labels=['no default', 'default'])
    display_cf.plot()
    plot1 = plt.figure(1)
    plot2 = plt.figure(2)
    tree.plot_tree(model_tree, fontsize= 7, filled=True, feature_names=X.columns, class_names=['no default', 'default'])
    plt.show()


def main():
    StoreAnalysis()
    DefaultDecisionTree()

if __name__ == '__main__':
    pd.set_option('display.width', None)
    pd.set_option('display.max_columns', None)

    main()