#!/usr/bin/env python
# coding=utf-8
def forward_replacement(embeddings, batch_indexes, poisoning_indexes, target_indexes, blurred=False):
    replacement_map = []
    poisoning_set = []
    target_set = []
    
    for i, bindex in enumerate(batch_indexes):
        if bindex in poisoning_indexes:
            poisoning_set.append(i)
        if (bindex in target_indexes) and (bindex not in poisoning_indexes):
            target_set.append(i)
    
    if blurred:
        for i in poisoning_set:
            embeddings[i] = torch.randn(embeddings[i].shape).to(embeddings[i].device)
    
    if len(target_set) > 0 and len(poisoning_set) > 0:
        for i in target_set:
            j = np.random.choice(poisoning_set)
            replacement_map.append((i, j))
            embeddings[i] = embeddings[j]
    
    return embeddings, replacement_map

def backward_replacement(gradients, replacement_map, gamma):
    for (i, j) in replacement_map:
        gradients[j] = gamma * gradients[i]

    return gradients
