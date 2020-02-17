#encoding: utf-8

from sklearn.ensemble._base import _BaseHeterogeneousEnsemble
from sklearn.metrics import accuracy_score
from math import sqrt

class Metodo_acuracia(_BaseHeterogeneousEnsemble):
    def __init__(self, estimators, numPontos, accMin, X, y):
        self.estimators = estimators
        self.numPontos = numPontos
        self.accMin = accMin
        self.base = list(zip(X, y))#armazena objetos usados para escolher estimadores
        
    def fit(self, X, y):
        for estimator in self.estimators:
            estimator.fit(X, y)
    
    def predict(self, X):
        return [self._predict(x) for x in X]
            
    def _predict(self, x):
        """Recebe objeto x
           Retorna predição y
        """
        selectedEstimators = self._selectEstimators(x)
        
        return self._weightedVoting(selectedEstimators, x)
    

    def _weightedVoting(self, weight_estimators, x):
        """Recebe lista de tuplas (peso, estimador) e objeto x
           Retorna classe escolhida
        """
        votes = dict()
        
        for w, estimator in weight_estimators:
            y_pred = estimator.predict([x])[0]
            if y_pred in votes.keys():
                newValue = votes.get(y_pred) + w
                votes.update({y_pred : newValue})
            else:
                votes.update({y_pred : w})
                
        winner = sorted(votes, reverse = True)[0]
        
        return winner
        

    def _selectEstimators(self, x):
        """Recebe um objeto x
           Retorna lista de tuplas (peso, estimador)
        """
        X, y_real = zip(*self._getNearbyPoints(x))
        
        estimatorsAccuracies = [(
            accuracy_score(y_real, estimator.predict(X)),
            estimator
            ) for estimator in self.estimators
        ]
        
        selectedEstimators = list(filter(lambda est: est[0]>self.accMin, estimatorsAccuracies))
        
        #caso nenhum estimador seja escolhido, seleciona o com melhor acurácia
        if len(selectedEstimators)==0:
            bestEstimator = sorted(estimatorsAccuracies, reverse = True, key = lambda accEst: accEst[0])[0]
            selectedEstimators = [bestEstimator]
        
        accTotal = sum([acc for acc, _ in selectedEstimators])
        
        weightedEstimators = [(
            acc/accTotal, 
            estimator
            ) for acc, estimator in selectedEstimators]
        
        return weightedEstimators

    def _getNearbyPoints(self, x):
        """Recebe um objeto x
           Retorna objetos  de self.base mais próximos de x
        """

        pointsSortedByDistance = sorted(self.base, key= lambda obj: self._distance(obj[0], x))
        
        return pointsSortedByDistance[:self.numPontos]
        
    def _distance(self, obj1, obj2):
        """Recebe dois objetos e retorna distância euclidiana entre estes"""
        return sqrt(sum([(x - y)**2 for x, y in zip(obj1, obj2)]))

class Metodo_similaridade(_BaseHeterogeneousEnsemble):
    def _init_(self, estimators, numPontos, qtdeClassificadores):
        self.estimators = estimators    
        self.numPontos = numPontos
        self.qtdeClassificadores = qtdeClassificadores
        
    def fit(self, X, y):
        #TODO com base em X fazer matriz de covariância
        for estimator in self.estimators:
            estimator.fit(X, y)
    
    def predict(self, X):
        return [self._predict(x) for x in X]
            
    def _predict(self, x):
        selectedEstimators = self._selectEstimators(x)
        return self._voting(selectedEstimators, x)
        
    def _createObjects(self):
        pass
        
    def _selectEstimators(self, x):
        pass
        
    def _voting(self, estimators, x):
        pass
