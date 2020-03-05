from classificadores import Metodo_acuracia, Metodo_similaridade
from sklearn.dummy import DummyClassifier
from math import sqrt

class Classifier:
    def __init__(self, prediction):
        self.prediction = prediction
    def predict(self, x):
        return [self.prediction]

def test_weightedVoting():
    
    macc = Metodo_acuracia(0, 0, 0, [0], [0])

    clfs = [ Classifier('a'), Classifier('a'), Classifier('b'), Classifier('c')]
    weights = [0.2, 0.1, 0.5, 0.2]
    weightsEstimators = zip(weights, clfs)
    
    assert 'b' == macc._weightedVoting(weightsEstimators, 0)
    
def test_voting():
    
    msim = Metodo_similaridade(0, 0, 0)

    clfs = [ Classifier('a'), Classifier('a'), Classifier('b'), Classifier('c')]
    
    assert 'a' == msim._voting(clfs, 0)
    
def test_decisionSimilarity():

    msim = Metodo_similaridade(0, 0, 0)

    y_preds = [1, 1, 0, 1, 0]
    Y = [[1, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [1, 0, 0, 0, 0]]
    
    assert 8 == msim._decisionSimilarity(y_preds, Y)
    
def test_getNearbyPoints():

    X = [[10, 2, -5],
        [2, 2, 2],
        [0, 0, 1],
        [-10, 0, 0],
        [2, 0, 0],
        [50, 2, -2]]
    y = [0, 0, 0, 0, 0, 0]
    
    x = [0, 0, 0]
    numObjects = 3

    nearest = [([0, 0, 1], 0),([2, 0, 0], 0),([2, 2, 2], 0)]

    macc = Metodo_acuracia(0, numObjects, 0, X, y)
    
    assert nearest == macc._getNearbyPoints(x)
    
def test_distance():

    macc = Metodo_acuracia(0, 0, 0, [0], [0])
    x = [1, 1, 1]
    y = [2, 1, 0]
    assert sqrt(2) == macc._distance(x, y)
    
