from classificadores import Metodo_acuracia, Metodo_similaridade
from sklearn.dummy import DummyClassifier


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
