from collections import Counter

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
