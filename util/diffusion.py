import torch
from layers.normalization import L2N
import numpy as np
from sklearn import preprocessing


def alpha_query_expansion(vecs, alpha, n):
    scores = torch.mm(vecs, vecs.T)
    ranks = torch.argsort(-scores, dim=1)
    newvecs = torch.zeros_like(vecs)
    for i in range(vecs.shape[0]):
        nqe = ranks[i, 1:n+1]
        weights = scores[i,nqe]
        weights = torch.pow(weights, alpha)
        newvecs[i] = torch.sum(weights.expand(vecs.shape[1], -1).T * vecs[nqe], axis=0)
     
    newvecs = preprocessing.normalize(newvecs, norm='l2', axis=1)

    return newvecs


class kReciprocalReRanking(object):
    def __init__(self, query_features, base_features, cosine=False):
        self.q_b_dist = np.dot(query_features, base_features.T)
        self.q_q_dist = np.dot(query_features, query_features.T)
        self.b_b_dist = np.dot(base_features, base_features.T)

        original_dist = np.concatenate(
            [np.concatenate([self.q_q_dist, self.q_b_dist], axis=1),
             np.concatenate([self.q_b_dist.T, self.b_b_dist], axis=1)],
            axis=0)
        if cosine:
            self.original_dist = original_dist
        else:
            original_dist = 2. - 2 * original_dist  # change the cosine similarity metric to euclidean similarity metric
            original_dist = np.power(original_dist, 2).astype(np.float32)
            self.original_dist = np.transpose(1. * original_dist / np.max(original_dist, axis=0))
        # initial_rank = np.argsort(original_dist).astype(np.int32)
        # top K1+1

    def forward(self, k1=20, k2=6, l=0.3):
        initial_rank = np.argpartition(self.original_dist, range(1, k1 + 1))
        query_num = self.q_b_dist.shape[0]
        all_num = self.original_dist.shape[0]
        V = np.zeros_like(self.original_dist).astype(np.float32)
        for i in range(all_num):
            # k-reciprocal neighbors
            k_reciprocal_index = self.k_reciprocal_neigh(initial_rank, i, k1)
            k_reciprocal_expansion_index = k_reciprocal_index
            for j in range(len(k_reciprocal_index)):
                candidate = k_reciprocal_index[j]
                candidate_k_reciprocal_index = self.k_reciprocal_neigh(initial_rank, candidate, int(np.around(k1 / 2)))
                if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2. / 3 * len(
                        candidate_k_reciprocal_index):
                    k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

            k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
            weight = np.exp(-self.original_dist[i, k_reciprocal_expansion_index])
            V[i, k_reciprocal_expansion_index] = 1. * weight / np.sum(weight)

        original_dist = self.original_dist[:query_num, ]
        if k2 != 1:
            V_qe = np.zeros_like(V, dtype=np.float32)
            for i in range(all_num):
                V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
            V = V_qe
            del V_qe
        del initial_rank
        invIndex = []
        for i in range(all_num):
            invIndex.append(np.where(V[:, i] != 0)[0])

        jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)

        for i in range(query_num):
            temp_min = np.zeros(shape=[1, all_num], dtype=np.float32)
            indNonZero = np.where(V[i, :] != 0)[0]
            indImages = [invIndex[ind] for ind in indNonZero]
            for j in range(len(indNonZero)):
                temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                                   V[indImages[j], indNonZero[j]])
            jaccard_dist[i] = 1 - temp_min / (2. - temp_min)

        final_dist = jaccard_dist * (1 - l) + original_dist * l
        del original_dist
        del V
        del jaccard_dist
        final_dist = final_dist[:query_num, query_num:]
        return final_dist

    @staticmethod
    def k_reciprocal_neigh(initial_rank, i, k1):
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        return forward_k_neigh_index[fi]


