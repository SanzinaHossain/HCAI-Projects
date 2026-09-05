import numpy as np
from scipy.optimize import minimize


def _sigmoid(z):
    z = np.clip(z, -30, 30)
    return 1/(1+np.exp(-z))


def fit_bradley_terry(X, choices, l2=1.0):
    d = X.shape[1]
    if not choices: return np.zeros(d)
    diffs = np.asarray([X[a]-X[b] for a,b in choices])
    def obj(w):
        p = np.clip(_sigmoid(diffs @ w), 1e-9, 1-1e-9)
        return -np.log(p).sum() + 0.5*l2*np.dot(w,w)
    return minimize(obj, np.zeros(d), method='L-BFGS-B').x


def fit_plackett_luce(X, rankings, l2=1.0):
    d = X.shape[1]
    if not rankings: return np.zeros(d)
    def obj(w):
        loss = 0.5*l2*np.dot(w,w)
        for ranking in rankings:
            remaining = list(ranking)
            while len(remaining) > 1:
                chosen = remaining[0]
                util = X[remaining] @ w
                m = util.max()
                logden = m + np.log(np.exp(util-m).sum())
                loss += -(X[chosen] @ w - logden)
                remaining.pop(0)
        return loss
    return minimize(obj, np.zeros(d), method='L-BFGS-B').x


def pairwise_accuracy(w, X, validation):
    if not validation: return None
    correct = 0
    for winner, loser in validation:
        correct += int((X[winner] @ w) >= (X[loser] @ w))
    return correct/len(validation)


def recommend(w, X, df, exclude_ids=None, n=5):
    exclude_ids = set(exclude_ids or [])
    scores = X @ w
    order = np.argsort(-scores)
    out=[]
    for idx in order:
        if int(idx) in exclude_ids: continue
        out.append((int(idx), float(scores[idx])))
        if len(out)>=n: break
    return out
