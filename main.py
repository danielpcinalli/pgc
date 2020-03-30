#encoding: utf-8

from classificadores import Metodo_acuracia, Metodo_similaridade
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from datasets import load_ecoli, load_anuran


def main():
    numPontosProximos = 10
    accMin = 0.9
    #testeMetodoAcuracia(numPontosProximos, accMin)
    
    numObjetosGerados = 10
    for qtdeClassificadores in [5, 15, 80]:
        testeMetodoSimilaridade(numObjetosGerados, qtdeClassificadores)

def testeMetodoAcuracia(numPontosProximos, accMin):
    
    print("MÉTODO: ACURÁCIA")
    print(f"Quantidade de objetos próximos: {numPontosProximos}")
    print(f"Acurácia mínima: {accMin}")
    X, y = load_anuran()

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 10)
    X_train, X_base, y_train, y_base = train_test_split(X_train, y_train, random_state = 100)
    
    #clfs = [DecisionTreeClassifier() for i in range(100)]
    clfs = [RandomForestClassifier(n_estimators = 30) for i in range(20)]
    
    macc = Metodo_acuracia(clfs, numPontosProximos, accMin, X_base, y_base)
    
    macc.fit(X_train, y_train)
    y_pred = macc.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia : {acc}")
    
    
def testeMetodoSimilaridade(numObjetosGerados, qtdeClassificadores):

    print("MÉTODO: SIMILARIDADE")
    print(f"Quantidade de objetos gerados: {numObjetosGerados}")
    print(f"Quantidade de classificadores escolhidos: {qtdeClassificadores}")
    
    X, y = load_anuran()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 10)
    
    #clfs = [DecisionTreeClassifier() for i in range(100)]
    clfs = [RandomForestClassifier(n_estimators = 20) for i in range(80)]

    msim = Metodo_similaridade(clfs, numObjetosGerados, qtdeClassificadores)
    msim.fit(X_train, y_train)
    y_pred = msim.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia : {acc}")









if __name__ == "__main__":
    main()
