Decision Trees
==============

With the help of Secret Sharing, a secure multi-party computation technique,
SecretFlow implements provably secure gradient boosting model
:py:meth:`~secretflow.ml.boost.ss_xgb_v.model.Xgb`
to support both regression and binary classification machine learning tasks.

Dataset Settings
----------------
vertically partitioned dataset:

- samples are aligned among the participants
- different participant obtains different features
- one participant owns the label

.. image:: resources/v_dataset.png


XGBoost Training Algorithm
--------------------------
Algorithm details can be found in `the official documents <https://xgboost.readthedocs.io/en/stable/tutorials/model.html>`_.
The main process of building a single tree is as follows:

- Statistics calculating: calculate the first-order gradient :math:`g_{i}` and second-order gradient :math:`h_{i}`
  for each sample with current prediction and label, according to the definition of loss function.

- Node splitting: enumerates all possible split candidates and choose the best one with the maximal gain.
  A split candidate is consisted of a split feature and a split value, which divides the samples in current node
  :math:`I` into two child nodes :math:`I_{L}` and :math:`I_{R}`, according to their feature values. Then, a split
  gain is computed with the following formula:

  .. image:: resources/gain_formula.png
      :height: 120px
      :width: 992px
      :scale: 50 %

  where :math:`\lambda` and :math:`\gamma` are the regularizers for the leaf number and leaf weights respectively.
  In this way, we can split the nodes recursively until the leaf.


- Weight calculating: calculate the weights of leaf nodes with the following formula:

  .. image:: resources/weight_formula.png
      :height: 138px
      :width: 382px
      :scale: 45 %

Regression and classification share the same training process except:

1. they employs different loss functions, i.e. MSE for regression and Logloss for classification.
2. classification executes an extra sigmoid function to transform the prediction into a probability.

SS-XGB Training
---------------
SS-XGB :py:meth:`~secretflow.ml.boost.ss_xgb_v.model.Xgb` use secret sharing to compute the split gain and leaf weights.

In order to implement a secure joint training, we replace all the computations with secret sharing protocols,
e.g. Addition, Multiplication, etc. In addition, we have to take special care to accumulate the gradients
without leaking out the feature partial order of samples.

This problem can be solved by introducing an indicator vector 𝑆.

.. image:: resources/indicator_vecto.jpg

The samples to be accumulated is marked as 1 in 𝑆 and 0 otherwise. To preserve privacy, the indicator vector also
transformed to secret shares. In this way, the sum of the gradients of the samples can be computed as the inner
product of the indicator vector and the gradient vector, which can be securely computed by secret sharing protocols.

Similarly, the indicator trick can be used to hide the instance distribution on nodes. Refer to our paper
`Large-Scale Secure XGB for Vertical Federated Learning <https://arxiv.org/pdf/2005.08479.pdf>`_
for more details about SS-XGB algorithm and security analysis.

Example
--------

A local cluster(Standalone Mode) needs to be initialized as the running environment for this example.
See `Deployment <../../getting_started/deployment>`_ and refer to the 'Cluster Mode'.

For more details about the APIs, see :py:meth:`~secretflow.ml.boost.ss_xgb_v.model.Xgb`

.. code-block:: python

    import sys
    import time
    import logging

    import secretflow as sf
    from secretflow.ml.boost.ss_xgb_v import Xgb
    from secretflow.device.driver import wait, reveal
    from secretflow.data import FedNdarray, PartitionWay
    from secretflow.data.split import train_test_split
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import accuracy_score, classification_report


    # init log
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # init all nodes in local Standalone Mode.
    sf.init(['alice', 'bob', 'carol'], address='local')

    # init PYU, the Python Processing Unit, process plaintext in each node.
    alice = sf.PYU('alice')
    bob = sf.PYU('bob')
    carol = sf.PYU('carol')

    # init SPU, the Secure Processing Unit,
    #           process ciphertext under the protection of a multi-party secure computing protocol
    spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob', 'carol']))

    # read data in each party
    def read_x(start, end):
        from sklearn.datasets import load_breast_cancer
        x = load_breast_cancer()['data']
        return x[:, start:end]

    def read_y():
        from sklearn.datasets import load_breast_cancer
        return load_breast_cancer()['target']

    # alice / bob / carol each hold one third of the features of the data
    v_data = FedNdarray(
        partitions={
            alice: alice(read_x)(0, 10),
            bob: bob(read_x)(10, 20),
            carol: carol(read_x)(20, 30),
        },
        partition_way=PartitionWay.VERTICAL,
    )
    # Y label belongs to alice
    label_data = FedNdarray(
        partitions={alice: alice(read_y)()},
        partition_way=PartitionWay.VERTICAL,
    )
    # wait IO finished
    wait([p.data for p in v_data.partitions.values()])
    wait([p.data for p in label_data.partitions.values()])
    # split train data and test data
    random_state = 1234
    split_factor = 0.8
    v_train_data, v_test_data = train_test_split(v_data, train_size=split_factor, random_state=random_state)
    v_train_label, v_test_label= train_test_split(label_data, train_size=split_factor, random_state=random_state)
    # run SS-XGB
    xgb = Xgb(spu)
    start = time.time()
    params = {
        # for more detail, see Xgb API doc
        'num_boost_round': 5,
        'max_depth': 5,
        'learning_rate': 0.1,
        'sketch_eps': 0.08,
        'objective': 'logistic',
        'reg_lambda': 0.1,
        'subsample': 1,
        'colsample_by_tree': 1,
        'base_score': 0.5,
    }
    model = xgb.train(params, v_train_data,v_train_label)
    logging.info(f"train time: {time.time() - start}")

    # Do predict
    start = time.time()
    # Now the result is saved in the spu by ciphertext
    spu_yhat = model.predict(v_test_data)
    # reveal for auc, acc and classification report test.
    yhat = reveal(spu_yhat)
    logging.info(f"predict time: {time.time() - start}")
    y = reveal(v_test_label.partitions[alice])
    # get the area under curve(auc) score of classification
    logging.info(f"auc: {roc_auc_score(y, yhat)}")
    binary_class_results = np.where(yhat>0.5, 1, 0)
    # get the accuracy score of classification
    logging.info(f"acc: {accuracy_score(y, binary_class_results)}")
    # get the report of classification
    print("classification report:")
    print(classification_report(y, binary_class_results))
