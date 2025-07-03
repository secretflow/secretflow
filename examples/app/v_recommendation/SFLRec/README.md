# Introduction
This project proposes a novel **split federated learning** framework. We integrates heterogeneous graph attention networks and knowledge distillation to address critical challenges including data heterogeneity, data silos, and privacy preservation.

# Dataset
Download the [douban-moive datasets](https://www.kaggle.com/datasets/fengzhujoey/douban-datasetratingreviewside-information/code). 

The SFL Recommendation system trains on the output embeddings of the heterogeneous graph classifier.

# Heterogeneous graph classifier(SplitHAN-douban.ipynb)
To address data heterogeneity challenges, the hierarchical attention mechanism (including node-level and semantic-level attention) is employed to classify and train the heterogeneous graph data. Specifically, you need to design **meta-paths** based on data characteristics. The mechanism learns the importance between nodes and their meta-path-based neighbors and aggregates multi-semantic information to generate node embeddings for subsequent split federated recommendation tasks.

To mitigate risks of raw data leakage during transmission, the framework adopts split learning for privacy-preserving model training.

## Technical Advantages:

- Supports federated training across isolated data silos while maintaining GDPR-compliant privacy standards.
- Obtain 88% accuracy on classification tasks.


# Splited Federated Learning (combine-GNN-sf.py)
The Split Federated Recommendation System Framework proposed in this study adopts a two-client architecture with a central server, designed to balance computational efficiency and privacy preservation. Below is the technical breakdown of the framework based on the described workflow:

## Architectural Components
### Clients (A & B):

- **Privacy Model**: Maintained locally using on-device data to ensure sensitive information (e.g., user preferences, interaction logs) never leaves the client.
- **Public Model**: Partitioned into three segments:
  - Head Layers (Client A): Handles initial data processing (e.g., embedding user behavior sequences, text tokenization) and lightweight feature extraction.
  - Intermediate Layers (Server): Performs compute-intensive operations like cross-client feature interaction modeling (e.g., graph convolution, attention-based fusion) and high-dimensional tensor transformations.
  - Tail Layers (Client B): Generates final recommendations (e.g., ranking scores, CTR prediction) and computes gradients during backpropagation.

### Server:
- Aggregates intermediate outputs from clients using federated averaging (FedAvg).
- Broadcasts updated public model parameters to both clients synchronously.

# Knowledge Distillation

To addresse the challenge of leveraging non-shared data by incorporating knowledge distillation into the federated recommendation framework. The framework establishes the public model as the teacher model and client-side privacy models (often lightweight neural networks) as student models. During training, both KL divergence and distillation loss are integrated into the objective function to quantify the alignment between the outputs of these models. 
