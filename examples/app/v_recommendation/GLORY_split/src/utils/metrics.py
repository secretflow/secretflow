import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F


def area_under_curve(y_true, y_hat):
    """
    Function for calculating the auc.
    Inputs:
        y_true - True labels
        y_hat - Predicted labels
    Outputs:
        auc - auc between the predicted and real labels
    """

    y_hat = torch.argmax(y_hat, dim=-1)
    tot = y_true.shape[0]
    hit = torch.sum(y_true == y_hat)
    return hit.data.float() * 1.0 / tot


def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def ctr_score(y_true, y_score, k=1):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    return np.mean(y_true)


def cal_metric(pair):
    auc = roc_auc_score(*pair)
    mrr = mrr_score(*pair)
    ndcg5 = ndcg_score(*pair, 5)
    ndcg10 = ndcg_score(*pair, 10)
    return  auc, mrr, ndcg5, ndcg10


# Diversity Metrics

def ILAD(vecs):
    # similarity = torch.mm(vecs, vecs.T)
    # similarity = cosine_similarity(X=vecs)
    similarity = F.cosine_similarity(vecs.unsqueeze(dim=0), vecs.unsqueeze(dim=1))
    distance = (1 - similarity)/2
    score = distance.mean()-1/distance.shape[0]
    return score.item()


def ILMD(vecs):
    # similarity = torch.mm(vecs, vecs.T)
    # similarity = cosine_similarity(X=vecs)
    similarity = F.cosine_similarity(vecs.unsqueeze(dim=0), vecs.unsqueeze(dim=1))
    distance = (1 - similarity) / 2
    score = distance.min()
    return score.item()

def density_ILxD(scores, news_emb, top_k=5):
    """
    Args:
        scores: [batch_size, y_pred_score]
        news_emb: [batch_size, news_num, news_emb_size]
        top_k: integer, n=5, n=10
    """
    top_ids = torch.argsort(scores)[-top_k:]
    news_emb =  news_emb / torch.sqrt(torch.square(news_emb).sum(dim=-1)).reshape((len(news_emb), 1))
    # nv: (top_k, news_emb_size)
    nv = news_emb[top_ids]
    ilad = ILAD(nv)
    ilmd = ILMD(nv)

    return ilad, ilmd




