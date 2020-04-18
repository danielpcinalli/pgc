#encoding: utf-8

from sklearn.ensemble._base import _BaseHeterogeneousEnsemble
from sklearn.metrics import accuracy_score
from math import sqrt
import numpy as np
import util
from collections import Counter

class Metodo_acuracia(_BaseHeterogeneousEnsemble):
    def __init__(self, estimators, numObjects, accMin):
        self.estimators = estimators
        self.numObjects = numObjects
        self.accMin = accMin

    def setKnownObjects(self, X, y):
        self.base = list(zip(X, y))
        if self.numObjects > len(y):
            self.numObjects = (int)(len(y) * 0.1)
            print(f"Número de objetos próximos selecionados maior que quantidade de objetos conhecidos, alterado para {self.numObjects}")

    def fit(self, X, y):
        self.setKnownObjects(X, y)
        for estimator in self.estimators:
            estimator.fit(X, y)

    def predict(self, X):
        if self.numObjects == len(self.base):
            return [self._predict_full(x) for x in X]
        return [self._predict(x) for x in X]


    def _predict_full(self, x):
        """Realiza predição utilizando todos os estimadores, sem seleção, apenas para comparação"""
        y_preds = [estimator.predict([x])[0] for estimator in self.estimators]
        return util.voting(y_preds)

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
        weights_preds = [(w, estimator.predict([x])[0]) for w, estimator in weight_estimators]
        return util.weightedVoting(weights_preds)


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
        if len(selectedEstimators) == 0:
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

        return pointsSortedByDistance[:self.numObjects]

    def _distance(self, obj1, obj2):
        """Recebe dois objetos e retorna distância euclidiana entre estes"""
        return sqrt(sum([(x - y)**2 for x, y in zip(obj1, obj2)]))

class Metodo_similaridade(_BaseHeterogeneousEnsemble):
    def __init__(self, estimators, numObjects, qtdeClassificadores):
        self.estimators = estimators
        self.numObjects = numObjects
        self.qtdeClassificadores = qtdeClassificadores

    def fit(self, X, y):

        #usando desvio padrão para usar np.random.normal
        self.sigmas = [sqrt(np.var(column)/10) for column in X.T]

        for estimator in self.estimators:
            estimator.fit(X, y)

    def predict(self, X):
        if self.qtdeClassificadores == len(self.estimators):
            return [self._predict_full(x) for x in X]
        return [self._predict(x) for x in X]

    def _predict(self, x):
        selectedEstimators = self._selectEstimators(x)
        return self._voting(selectedEstimators, x)

    def _predict_full(self, x):
        """Realiza predição utilizando todos os estimadores, sem seleção, apenas para comparação"""
        return self._voting(self.estimators, x)

    def _createObjects(self, x):
        objects = [np.random.normal(x, self.sigmas) for _ in range(self.numObjects)]
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
        votes = Counter()

        y_preds = [estimator.predict([x])[0] for estimator in estimators]
        return util.voting(y_preds)

    def _decisionSimilarity(self, y_preds, Y):
        """Recebe array de predições de um estimador e lista de array de predições de todos os estimadores
        Retorna similaridade de decisão
        """

        decisionSimilarity = 0
        for y in Y:
            decisionSimilarity += [a == b for a, b in zip(y, y_preds)].count(True)

        return decisionSimilarity
