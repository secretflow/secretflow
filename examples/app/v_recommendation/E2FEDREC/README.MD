# E2Rec: Dual-Augmentation Based Cross-Domain Recommendation

**E2Rec** is a contrastive-learning-based cross-domain recommendation framework designed to tackle data sparsity and domain discrepancy problems. It leverages both **intra-domain** and **inter-domain** contrastive learning to enhance user and item embeddings and improve recommendation performance across domains.

---

## 🔍 Key Innovations

Traditional cross-domain recommendation models struggle with:
- Limited overlapping users between domains;
- Significant differences in user behaviors and item semantics;
- Ineffective representation transfer across heterogeneous domains.

To address these challenges, **E2Rec** introduces a dual-augmentation framework:
- **Intra-domain contrastive learning** to improve local embedding quality.
- **Inter-domain contrastive learning** to align user embeddings across domains.

---

## 🧠 Technical Architecture

E2Rec includes two main modules and one joint training objective:

### 1. Intra-domain Contrastive Learning

This module enhances representation discrimination within each domain:

- Uses clustering (e.g., K-Means) to group users/items and construct prototype embeddings.
- Forms positive and negative contrastive pairs to increase intra-class compactness and inter-class separation.
- Applies temperature scaling to control the hardness of samples and training stability.

### 2. Inter-domain Contrastive Learning

This module aligns user representations across domains:

- Builds cross-domain prototypes based on overlapping users.
- Applies inter-domain contrastive loss to minimize distribution gaps.
- Enhances representation consistency between domains for better transferability.

### 3. Multi-task Joint Optimization

The final objective combines the supervised recommendation loss with both contrastive losses:

$$
L = L_{rec} + \lambda_{intra} L_{intra} + \lambda_{inter} L_{inter}
$$

---

## Train

To train E2FCDR, you can run the following command:

```bash
python -u main.py \
    --maxEpochs 20 \
    --batchSize 512 \
    --lr 0.001