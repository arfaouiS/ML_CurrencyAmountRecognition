from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB, MultinomialNB


def svmModel(X_train, y_train, param_grid):
    svm = SVC()
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


def naiveBayesModel(X_train, y_train, param_grid):
    nb = GaussianNB()
    grid_search = GridSearchCV(nb, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


def kNeighborsClassifier(X_train, y_train, param_grid):
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


def decisionTreeClassifier(X_train, y_train, param_grid):
    tree = DecisionTreeClassifier()
    grid_search = GridSearchCV(tree, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


