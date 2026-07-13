import numpy as np

def k_reciprocal_re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):
    num_q = q_g_dist.shape[0]
    num_g = q_g_dist.shape[1]
    num_total = num_q + num_g
    
    original_dist = np.concatenate(
        [np.concatenate([q_q_dist, q_g_dist], axis=1),
         np.concatenate([q_g_dist.T, g_g_dist], axis=1)], axis=0
    )
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = original_dist / np.max(original_dist, axis=0)
    
    initial_rank = np.argpartition(original_dist, range(1, k1 + 1), axis=1)
    nn_k1 = initial_rank[:, :k1]
    nn_k1_half = initial_rank[:, :int(np.ceil(k1 / 2))]
    
    V = np.zeros((num_total, num_total), dtype=np.float32)
    for i in range(num_total):
        k_reciprocal_index = []
        for j in nn_k1[i]:
            if i in nn_k1[j]:
                k_reciprocal_index.append(j)
                
        for j in nn_k1_half[i]:
            if j not in k_reciprocal_index:
                k_reciprocal_candidates = []
                for candidate in nn_k1[j]:
                    if j in nn_k1[candidate]:
                        k_reciprocal_candidates.append(candidate)
                if len(np.intersect1d(k_reciprocal_index, k_reciprocal_candidates)) > 2/3 * len(k_reciprocal_candidates):
                    k_reciprocal_index.append(j)
                    
        k_reciprocal_index = np.unique(k_reciprocal_index)
        weight = np.exp(-original_dist[i, k_reciprocal_index])
        V[i, k_reciprocal_index] = weight / np.sum(weight)
    
    original_dist = original_dist[:num_q, num_q:]
    if k2 > 1:
        V_q = V[:num_q, :]
        V_g = V[num_q:, :]
        
        re_rank_dist = np.zeros_like(original_dist)
        for i in range(num_q):
            query_neighbors = np.where(V_q[i, :] != 0)[0]
            for j in range(num_g):
                gallery_neighbors = np.where(V_g[j, :] != 0)[0]
                intersect = np.intersect1d(query_neighbors, gallery_neighbors)
                if len(intersect) == 0:
                    continue
                sum_q = np.sum(np.minimum(V_q[i, intersect], V_g[j, intersect]))
                sum_g = np.sum(np.maximum(V_q[i, intersect], V_g[j, intersect]))
                re_rank_dist[i, j] = 1 - sum_q / (sum_g + 1e-12)
                
        return lambda_value * original_dist + (1 - lambda_value) * re_rank_dist
    return original_dist
