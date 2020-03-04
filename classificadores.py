#encoding: utf-8

from sklearn.ensemble._base import _BaseHeterogeneousEnsemble
from sklearn.metrics import accuracy_score
from math import sqrt
import numpy as np

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
        
        winner = sorted(votes.items(), reverse = True, key = lambda v : v[1])[0][0]
        
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
    def __init__(self, estimators, numPontos, qtdeClassificadores):
        self.estimators = estimators    
        self.numPontos = numPontos
        self.qtdeClassificadores = qtdeClassificadores
        
    def fit(self, X, y):
        
        #usando desvio padrão para usar np.random.normal
        self.sigmas = [sqrt(np.var(column)/10) for column in X.T]
        
        for estimator in self.estimators:
            estimator.fit(X, y)
    
    def predict(self, X):
        return [self._predict(x) for x in X]
            
    def _predict(self, x):
        selectedEstimators = self._selectEstimators(x)
        return self._voting(selectedEstimators, x)
        
    def _createObjects(self, x):
        objects = [np.random.normal(x, self.sigmas) for _ in range(self.numPontos)]
        return objects
        
        
    def _selectEstimators(self, x):
        objects = self._createObjects(x)
        
        preds = [estimator.predict(objects) for estimator in self.estimators]
        decisionSimilarities = [(
            self._decisionSimilarity(pred, preds),
            estimator) for pred, estimator in zip(preds, self.estimators)]
            
        sortedEstimators = sorted(decisionSimilarities, reverse = True, key = lambda simEst: simEst[0])
        
        sortedEstimators = [estimator for _, estimator in sortedEstimators]
        
        return sortedEstimators[:self.qtdeClassificadores]
        
        
    def _voting(self, estimators, x):
        votes = dict()

        for estimator in estimators:
            y_pred = estimator.predict([x])[0]
            if y_pred in votes.keys():
                newValue = votes.get(y_pred) + 1
                votes.update({y_pred : newValue})
            else:
                votes.update({y_pred : 1})
                
        winner = sorted(votes.items(), reverse = True, key = lambda v : v[1])[0][0]
        
        return winner
        
    def _decisionSimilarity(self, y_preds, Y):
        """Recebe array de predições de um estimador e lista de array de predições de todos os estimadores
        Retorna similaridade de decisão 
        """
        
        decisionSimilarity = 0
        for y in Y:
            decisionSimilarity += [a == b for a, b in zip(y, y_preds)].count(True)
        
        return decisionSimilarity
        
        
        
