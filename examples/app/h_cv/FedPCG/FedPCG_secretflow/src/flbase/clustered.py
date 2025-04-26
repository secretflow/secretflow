'''
Reference:
https://github.com/Accenture/Labs-Federated-Learning/tree/clustered_sampling
'''
from itertools import product
from scipy.cluster.hierarchy import fcluster, linkage
from copy import deepcopy
from tqdm import tqdm
import torch
import numpy as np


class ClientSelection:
    def __init__(self, total, device="cpu"):
        self.total = total
        self.device = device

    def select(self, n, client_idxs, metric):
        pass

    def save_selected_clients(self, client_idxs, results):
        tmp = np.zeros(self.total)
        tmp[client_idxs] = 1
        tmp.tofile(results, sep=',')
        results.write("\n")

    def save_results(self, arr, results, prefix=''):
        results.write(prefix)
        np.array(arr).astype(np.float32).tofile(results, sep=',')
        results.write("\n")


'''Clustered Sampling Algorithm 1'''


class ClusteredSampling1(ClientSelection):
    def __init__(self, total, device, n_cluster):
        super().__init__(total, device)
        self.n_cluster = n_cluster

    def setup(self, n_samples):
        '''
        Since clustering is performed according to the clients sample size n_i,
        unless n_i changes during the learning process,
        Algo 1 needs to be run only once at the beginning of the learning process.
        '''
        epsilon = int(10 ** 10)
        client_ids = sorted(n_samples.keys())
        n_samples = np.array([n_samples[i] for i in client_ids])
        weights = n_samples / np.sum(n_samples)
        # associate each client to a cluster
        augmented_weights = np.array([w * self.n_cluster * epsilon for w in weights])
        ordered_client_idx = np.flip(np.argsort(augmented_weights))

        distri_clusters = np.zeros((self.n_cluster, self.total)).astype(int)
        k = 0
        for client_idx in ordered_client_idx:
            while augmented_weights[client_idx] > 0:
                sum_proba_in_k = np.sum(distri_clusters[k])
                u_i = min(epsilon - sum_proba_in_k, augmented_weights[client_idx])
                distri_clusters[k, client_idx] = u_i
                augmented_weights[client_idx] += -u_i
                sum_proba_in_k = np.sum(distri_clusters[k])
                if sum_proba_in_k == 1 * epsilon:
                    k += 1

        distri_clusters = distri_clusters.astype(float)
        for l in range(self.n_cluster):
            distri_clusters[l] /= np.sum(distri_clusters[l])

        self.distri_clusters = distri_clusters

    def select(self, n, client_idxs, metric=None):
        # selected_client_idxs = [int(np.random.choice(self.total, 1, p=self.distri_clusters[k])) for k in range(n)]
        selected_client_idxs = []
        for k in range(n):
            weight = np.take(self.distri_clusters[k], client_idxs)
            selected_client_idxs.append(int(np.random.choice(client_idxs, 1, p=weight / sum(weight))))
        return np.array(selected_client_idxs)


def add_and_div(samples):
    return samples / sum(samples)


## clients_dict[0]['num_train_samples']

'''Clustered Sampling Algorithm 2'''


class ClusteredSampling2(ClientSelection):
    def __init__(self, total, device, dist):
        super().__init__(total, device)
        self.distance_type = dist

    def setup(self, n_samples):
        """
        return the `representative gradient` formed by the difference
        between the local work and the sent global model
        """
        client_ids = sorted(n_samples.keys())
        n_samples = np.array([n_samples[i] for i in client_ids])
        print('n_samples', n_samples)
        self.weights = n_samples / sum(n_samples)

    #         self.weights = server_pyu(add_and_div)(n_samples)

    def init(self, global_m, local_models):
        self.prev_global_m = global_m
        self.gradients = self.get_gradients(global_m, local_models)

    def select(self, n, client_idxs, metric=None):
        # GET THE CLIENTS' SIMILARITY MATRIX
        sim_matrix = self.get_matrix_similarity_from_grads(
            self.gradients, distance_type=self.distance_type)
        # GET THE DENDROGRAM TREE ASSOCIATED
        linkage_matrix = linkage(sim_matrix, "ward")

        distri_clusters = self.get_clusters_with_alg2(linkage_matrix, n, self.weights)
        # sample clients
        selected_client_idxs = np.zeros(n, dtype=int)
        for k in range(n):
            selected_client_idxs[k] = int(np.random.choice(client_idxs, 1, p=distri_clusters[k]))
            # weight = np.take(distri_clusters[k], client_idxs)
            # selected_client_idxs[k] = int(np.random.choice(client_idxs, 1, p=weight/sum(weight)))
        # selected_client_idxs = np.take(client_idxs, selected_client_idxs)
        return selected_client_idxs

    def update(self, clients_models, sampled_clients_for_grad):
        print('>> update gradients')
        # UPDATE THE HISTORY OF LATEST GRADIENT
        gradients_i = self.get_gradients(self.prev_global_m, clients_models)
        for idx, gradient in zip(sampled_clients_for_grad, gradients_i):
            self.gradients[idx] = gradient

    def get_gradients(self, global_m, local_models):
        """
        return the `representative gradient` formed by the difference
        between the local work and the sent global model
        """
        local_model_params = []
        #         print('local_models',sf.reveal(local_models))
        for model in local_models:
            local_model_params += [
                [tens.detach().to(self.device) for tens in list(sf.reveal(model).parameters())][0]]  # .numpy()

        global_model_params = [tens.detach().to(self.device) for tens in list(sf.reveal(global_m).values())][0]

        local_model_grads = []
        for local_params in local_model_params:
            local_model_grads += [[local_weights - global_weights
                                   for local_weights, global_weights in
                                   zip(local_params, global_model_params)]]

        return local_model_grads

    def get_matrix_similarity_from_grads(self, local_model_grads, distance_type):
        """
        return the similarity matrix where the distance chosen to
        compare two clients is set with `distance_type`
        """
        n_clients = len(local_model_grads)

        # metric_matrix = np.zeros((n_clients, n_clients))
        metric_matrix = torch.zeros((n_clients, n_clients))
        for i, j in tqdm(product(range(n_clients), range(n_clients)), desc='>> similarity', ncols=80):
            metric_matrix[i, j] = self.get_similarity(
                local_model_grads[i], local_model_grads[j], distance_type)

        return metric_matrix

    def get_similarity(self, grad_1, grad_2, distance_type="L1"):
        if distance_type == "L1":
            norm = 0
            for g_1, g_2 in zip(grad_1, grad_2):
                # norm += np.sum(np.abs(g_1 - g_2))
                norm += torch.sum(torch.abs(g_1 - g_2))
            return norm.cpu().data

        elif distance_type == "L2":
            norm = 0
            for g_1, g_2 in zip(grad_1, grad_2):
                norm += np.sum((g_1 - g_2) ** 2)
            return norm

        elif distance_type == "cosine":
            norm, norm_1, norm_2 = 0, 0, 0
            print(grad_1[0].squeeze().shape, grad_2[0].squeeze().shape)
            for i in range(len(grad_1)):
                norm += np.sum(torch.mul(grad_1[i].squeeze(), grad_2[i].squeeze()))
                norm_1 += np.sum(grad_1[i] ** 2)
                norm_2 += np.sum(grad_2[i] ** 2)

            if norm_1 == 0.0 or norm_2 == 0.0:
                return 0.0
            else:
                norm /= np.sqrt(norm_1 * norm_2)
                return np.arccos(norm)

    def get_clusters_with_alg2(self, linkage_matrix: np.array, n_sampled: int, weights: np.array):
        """Algorithm 2"""
        epsilon = int(10 ** 10)

        # associate each client to a cluster
        link_matrix_p = deepcopy(linkage_matrix)
        augmented_weights = deepcopy(weights)

        for i in range(len(link_matrix_p)):
            idx_1, idx_2 = int(link_matrix_p[i, 0]), int(link_matrix_p[i, 1])

            new_weight = np.array(
                [sf.reveal(augmented_weights)[idx_1] + sf.reveal(augmented_weights)[idx_2]])
            augmented_weights = np.concatenate((augmented_weights, new_weight))
            link_matrix_p[i, 2] = int(new_weight * epsilon)

        clusters = fcluster(
            link_matrix_p, int(epsilon / n_sampled), criterion="distance")

        n_clients, n_clusters = len(clusters), len(set(clusters))

        # Associate each cluster to its number of clients in the cluster
        pop_clusters = np.zeros((n_clusters, 2), dtype=np.int64)
        for i in range(n_clusters):
            pop_clusters[i, 0] = i + 1
            for client in np.where(clusters == i + 1)[0]:
                pop_clusters[i, 1] += int(weights[client] * epsilon * n_sampled)

        pop_clusters = pop_clusters[pop_clusters[:, 1].argsort()]

        distri_clusters = np.zeros((n_sampled, n_clients), dtype=np.int64)

        # n_sampled biggest clusters that will remain unchanged
        kept_clusters = pop_clusters[n_clusters - n_sampled:, 0]

        for idx, cluster in enumerate(kept_clusters):
            for client in np.where(clusters == cluster)[0]:
                distri_clusters[idx, client] = int(
                    weights[client] * n_sampled * epsilon)

        k = 0
        for j in pop_clusters[: n_clusters - n_sampled, 0]:
            clients_in_j = np.where(clusters == j)[0]
            np.random.shuffle(clients_in_j)

            for client in clients_in_j:
                weight_client = int(weights[client] * epsilon * n_sampled)

                while weight_client > 0:
                    sum_proba_in_k = np.sum(distri_clusters[k])
                    u_i = min(epsilon - sum_proba_in_k, weight_client)
                    distri_clusters[k, client] = u_i
                    weight_client += -u_i
                    sum_proba_in_k = np.sum(distri_clusters[k])
                    if sum_proba_in_k == 1 * epsilon:
                        k += 1

        distri_clusters = distri_clusters.astype(float)
        print(distri_clusters.shape)
        for l in range(n_sampled):
            distri_clusters[l] /= np.sum(distri_clusters[l])

        return distri_clusters