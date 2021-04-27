from collections import Counter
import numpy as np
from operator import itemgetter
import math
from scipy.stats import rankdata
import pandas as pd

def weightedVoting(weight_votes):
    """Recebe lista de tuplas (peso, voto)"""
    counter = Counter()

    for w, v in weight_votes:
        counter[v] += w

    winner_votes_tuple = counter.most_common(1)[0]
    winner = winner_votes_tuple[0]

    return winner

def voting(votes):
    """Recebe lista votos"""
    counter = Counter()

    for v in votes:
        counter[v] += 1

    winner_votes_tuple = counter.most_common(1)[0]
    winner = winner_votes_tuple[0]

    return winner

def alert_sound():
    print('\a')

def bagging(X, y):
    size_of_bag = len(y)
    ids = np.random.choice(np.arange(size_of_bag), size_of_bag)
    bag_X = X[ids]
    bag_y = y[ids]
    return bag_X, bag_y

def getKSmallestIndexes(x, k):
    """
    Retorna índices dos k menores elementos de x
    """
    idxs = np.argpartition(x, k)
    return idxs[:k]

        

def getIndexedList(x, idxs):
    return list(itemgetter(*idxs)(x))

def distance(obj1, obj2):
    return math.sqrt(sum([(x - y)**2 for x, y in zip(obj1, obj2)]))

def rank_accuracy(df):
    """
    df: dataframe em que cada coluna é um algoritmo e cada índice um dataset, valores são acurácias
    Retorna dataframe com ranks
    """
    #rankdata considera menor valor como maior rank, portanto é necessário multiplicar por -1 as acurácias
    df = -1 * df
    ranks = pd.DataFrame(rankdata(df, axis=1), index=df.index, columns=df.columns)
    return ranks
