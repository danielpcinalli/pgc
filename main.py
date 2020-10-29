#encoding: utf-8
import traceback
from classificadores import Metodo_acuracia, Metodo_similaridade
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
#from sklearn import cross_validation
import sklearn
from datasets import load_leaf
import util

import numpy as np
np.random.seed(0)


def main():
    try:

        numPontosProximos = [10, 30, 50]
        accMin = 0.7
        for npp in numPontosProximos:
            testeMetodoAcuracia(npp, accMin)

        '''
        numObjetosGerados = 10
        for qtdeClassificadores in [5, 15, 25, 80]:
            testeMetodoSimilaridade(numObjetosGerados, qtdeClassificadores)
        '''
        util.alert_sound()
    except:
        traceback.print_exc()
        util.alert_sound()

def testeMetodoAcuracia(numPontosProximos, accMin):

    print("MÉTODO: ACURÁCIA")
    print(f"Quantidade de objetos próximos: {numPontosProximos}")
    print(f"Acurácia mínima: {accMin}")

    X, y = load_leaf()

    clfs = [DecisionTreeClassifier() for i in range(100)]
    #clfs = [RandomForestClassifier(n_estimators = 20) for i in range(50)]
    #clfs = [KNeighborsClassifier() for i in range(50)]

    macc = Metodo_acuracia(clfs, numPontosProximos, accMin)

    kf = KFold(n_splits=5, shuffle=True, random_state=1)

    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        macc_clone = sklearn.base.clone(macc)

        macc_clone.fit(X_train, y_train)
        y_pred = macc_clone.predict(X_test)

        accuracy = accuracy_score(y_pred, y_test)
        print(accuracy)

def testeMetodoSimilaridade(numObjetosGerados, qtdeClassificadores):

    print("MÉTODO: SIMILARIDADE")
    print(f"Quantidade de objetos gerados: {numObjetosGerados}")
    print(f"Quantidade de classificadores escolhidos: {qtdeClassificadores}")

    X, y = load_leaf()

    #clfs = [DecisionTreeClassifier() for i in range(100)]
    clfs = [RandomForestClassifier(n_estimators = 15) for i in range(80)]

    msim = Metodo_similaridade(clfs, numObjetosGerados, qtdeClassificadores)

    kf = KFold(n_splits=5, shuffle=True, random_state=1)

    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        msim_clone = sklearn.base.clone(msim)

        msim_clone.fit(X_train, y_train)
        y_pred = msim_clone.predict(X_test)

        accuracy = accuracy_score(y_pred, y_test)
        print(accuracy)



if __name__ == "__main__":
    main()
