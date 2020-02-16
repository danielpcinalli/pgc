#encoding: utf-8

from classificadores import Metodo_acuracia
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

load_dataset = load_breast_cancer
numPontosProximos = 10
accMin = 0.9

def main():

    X, y = load_dataset(return_X_y = True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 10)
    X_train, X_base, y_train, y_base = train_test_split(X_train, y_train, random_state = 100)
    
    #clfs = [DecisionTreeClassifier() for i in range(100)]
    clfs = [RandomForestClassifier(n_estimators = 30) for i in range(20)]
    
    macc = Metodo_acuracia(clfs, numPontosProximos, accMin, X_base, y_base)
    
    macc.fit(X_train, y_train)
    y_pred = macc.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia : {acc}")


if __name__ == "__main__":
    main()
