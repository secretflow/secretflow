Linear Models
==============

Linear model is a kind of statistical model with simple form and very widely used. 
Under the protection of a multi-party secure computing protocol, SecretFlow
implements provably secure linear regression and binary classification
regression through 
`batch Stochastic Gradient Descent (SGD) method <https://stats.stackexchange.com/questions/488017/understanding-mini-batch-gradient-descent>`_
for vertically partitioned dataset setting.

The matrix formula for batch Stochastic Gradient Descent (SGD) is as follows:

Normal:

:math:`{\theta^{t+1}} \leftarrow {\theta^t} - \frac{\alpha}{m}  {X}^T ({X}{\theta^t} - {y})`

L2 penalty:

:math:`{\theta^{t+1}} \leftarrow {\theta^t} - \frac{\alpha}{m}  ({X}^T ({X}{\theta^t} - {y}) + \lambda {w^t})`
where 
:math:`w^t_0 = 0, w^t_j = \theta^t_j`
:math:`j = 1, \cdots, n`

SecretFlow provides two provably security implementations of SGD:

- SS-SGD: SS-SGD is short for secret sharing SGD training, uses Secret Sharing to calculate the gradient.

- HESS-SGD: HESS-SGD is short for HE & secret sharing SGD training, uses homomorphic encryption to calculate the gradient.

Secret Sharing is sensitive to bandwidth and latency, and the homomorphic encryption scheme consumes more CPU power.

Secret Sharing can complete the modeling faster with LAN or 10 Gigabit network,
and with limited bandwidth and high latency network environment can use HE to improve the modeling speed.

The two implementations have basically the same logic and algorithm security settings other than gradient calculation.
According to the different CPU/bandwidth of the running environment, you can choose the appropriate implementation to use.


SS-SGD
-------
 
The SS-SGD module :py:meth:`~secretflow.ml.linear.ss_sgd.model.SSRegression`
provides both linear and logistic regression linear models
for vertical split dataset setting by using secret sharing with mini
batch SGD training solver.

`Linear regression <https://en.wikipedia.org/wiki/Linear_regression>`_
fits a linear model with coefficients w = (w1, ..., wp)
to minimize the residual sum of squares between the observed targets in
the dataset, and the targets predicted by the linear approximation.

`Logistic regression <https://en.wikipedia.org/wiki/Logistic_regression>`_
, despite its name, is a linear model for classification
rather than regression. logistic regression is also known in the literature
as logit regression, maximum-entropy classification (MaxEnt) or the log-linear
classifier. the probabilities describing the possible outcomes of a single trial
are modeled using a logistic function. This method can fit binary regularization
with optional L2 regularization.

Example
++++++++

A local cluster(Standalone Mode) needs to be initialized as the running environment for this example. 
See `Deployment <../../getting_started/deployment.html>`_ and refer to the 'Cluster Mode'.

For more detail about parameter settings, see API :py:meth:`~secretflow.ml.linear.ss_sgd.model.SSRegression`

.. code-block:: python

    import sys
    import time
    import logging

    import numpy as np
    import spu
    import secretflow as sf
    from secretflow.data.split import train_test_split
    from secretflow.device.driver import wait, reveal
    from secretflow.data import FedNdarray, PartitionWay
    from secretflow.ml.linear.ss_sgd import SSRegression

    from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

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
    # use breast_cancer as example
        from sklearn.datasets import load_breast_cancer
        from sklearn.preprocessing import StandardScaler
        x = load_breast_cancer()['data']
        # LR's train dataset must be standardized or normalized
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        return x[:, start:end]

    def read_y():
        from sklearn.datasets import load_breast_cancer
        return load_breast_cancer()['target']

    # alice / bob / carol each hold one third of the features of the data
    # read_x is execute locally on each node.
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
    # split train data and test date
    random_state = 1234
    split_factor = 0.8
    v_train_data, v_test_data = train_test_split(v_data, train_size=split_factor, random_state=random_state)
    v_train_label, v_test_label = train_test_split(label_data, train_size=split_factor, random_state=random_state)
    # run SS-SGD
    # SSRegression use spu to fit model.
    model = SSRegression(spu)
    start = time.time()
    model.fit(
        v_train_data,      # x
        v_train_label,  # y
        5,           # epochs
        0.3,         # learning_rate
        32,          # batch_size
        't1',        # sig_type
        'logistic',  # reg_type
        'l2',        # penalty
        0.1,         # l2_norm
    )
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
    binary_class_results = np.where(yhat > 0.5, 1, 0)
    # get the accuracy score of classification
    logging.info(f"acc: {accuracy_score(y, binary_class_results)}")
    # get the report of classification
    print("classification report:")
    print(classification_report(y, binary_class_results))


algorithm details
++++++++++++++++++
more detail for logistic regression:

Taking binary regression as an example, the main process is as follows:

    Step 1: Initialize the dataset

    - The data provider infeed their dataset into secret sharing and vertically concatenates them as X.
    - The data provide holds Y infeed it into Secret Sharing.
    - Initialize weights w to the initial value set in parameter under Secret Sharing.
    - X.rows must be greater than X.cols, otherwise: 1. model will not converge; 2. Y may leak.

    Step 2: Using mini-batch gradient descent, repeat the following steps until the target number of iterations is reached

    - Step 2.1: Calculate the predicted value: pred = sigmoid(batch_x * w). 
      The sigmoid can be approximated using Taylor expansion, piecewise function, inverse square sigmoid function, etc.
    - Step 2.2: Calculate: err = pred - y
    - Step 2.3: Calculate the gradient: grad = batch_x.transpose() * err
    - Step 2.4: If using L2 penalty, update gradient: grad = grad + w' * l2_norm, where the intercept term of w' is 0
    - Step 2.5: update weights: w = w - (grad * learning_rate / batch_size)

    Step 3: Output
    - At this time, weights w is in the secret sharing. You can output reveal (w) as plaintext or directly save the secret sharing as needed.

Security Analysis
++++++++++++++++++

The X/Y/W participating in the calculation are kept in the Secret Sharing from the beginning.
And there is no reveal operation in the calculation process,
so it is impossible to infer the information of the plaintext data through the interactive data in the calculation.

HESS-SGD
---------

The HESS-SGD module :py:meth:`~secretflow.ml.linear.hess_sgd.model.HESSLogisticRegression` implements provably
secure linear regression using homomorphic encryption and Secret Sharing.

The biggest difference from SS-SGD is that the gradient calculation which has the largest communication cost in SS-SGD
is replaced by locally homomorphic calculation implementation.
Due to the asymmetric nature of homomorphic encryption, currently HESS-SGD only supports 2PC.
The algorithm implementation reference is `<When Homomorphic Encryption Marries Secret Sharing:
Secure Large-Scale Sparse Logistic Regression and Applications
in Risk Control> <https://arxiv.org/pdf/2008.08753.pdf>`_,
and some engineering optimizations are carried out.

Example
++++++++

A local cluster(Standalone Mode) needs to be initialized as the running environment for this example. 
See `Deployment <../../getting_started/deployment.html>`_ and refer to the 'Cluster Mode'.

For more details about API, see :py:meth:`~secretflow.ml.linear.hess_sgd.model.HESSLogisticRegression`

.. code-block:: python

    import sys
    import time
    import logging

    import numpy as np
    import secretflow as sf
    from secretflow.data.split import train_test_split
    from secretflow.device.driver import wait, reveal
    from secretflow.data import FedNdarray, PartitionWay
    from secretflow.ml.linear.hess_sgd import HESSLogisticRegression

    from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

    # init log
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # init all nodes in local Standalone Mode. HESS-SGD only support 2PC
    sf.init(['alice', 'bob'], address='local')

    # init PYU, the Python Processing Unit, process plaintext in each node.
    alice = sf.PYU('alice')
    bob = sf.PYU('bob')

    # init SPU, the Secure Processing Unit,
    # process ciphertext under the protection of a multi-party secure computing protocol
    spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))

    # first, init a HEU device that alice is sk_keeper and bob is evaluator
    heu_config = sf.utils.testing.heu_config(sk_keeper='alice', evaluators=['bob'])
    heu_x = sf.HEU(heu_config, spu.cluster_def['runtime_config']['field'])

    # then, init a HEU device that bob is sk_keeper and alice is evaluator
    heu_config = sf.utils.testing.heu_config(sk_keeper='bob', evaluators=['alice'])
    heu_y = sf.HEU(heu_config, spu.cluster_def['runtime_config']['field'])

    # read data in each party
    def read_x(start, end):
    # use breast_cancer as example
        from sklearn.datasets import load_breast_cancer
        from sklearn.preprocessing import StandardScaler
        x = load_breast_cancer()['data']
        # LR's train dataset must be standardized or normalized
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        return x[:, start:end]

    def read_y():
        from sklearn.datasets import load_breast_cancer
        return load_breast_cancer()['target']

    # alice / bob  each hold half of the features of the data
    # read_x is execute locally on each node.
    v_data = FedNdarray(
        partitions={
            alice: alice(read_x)(0, 15),
            bob: bob(read_x)(15, 30),
        },
        partition_way=PartitionWay.VERTICAL,
    )
    # Y label belongs to bob
    label_data = FedNdarray(
        partitions={alice: alice(read_y)()},
        partition_way=PartitionWay.VERTICAL,
    )

    # wait IO finished
    wait([p.data for p in v_data.partitions.values()])
    wait([p.data for p in label_data.partitions.values()])
    # split train data and test date
    random_state = 1234
    split_factor = 0.8
    v_train_data, v_test_data = train_test_split(v_data, train_size=split_factor, random_state=random_state)
    v_train_label, v_test_label = train_test_split(label_data, train_size=split_factor, random_state=random_state)
    # run HESS-SGD
    # HESSLogisticRegression use spu / heu to fit model.
    model = HESSLogisticRegression(spu, heu_y, heu_x)
    # HESSLogisticRegression(spu, heu_x, heu_y)
    # spu – SPU SPU device.
    # heu_x – HEU HEU device without label.
    # heu_y – HEU HEU device with label.
    # Here, label belong to Alice(heu_x)
    start = time.time()
    model.fit(
        v_train_data,
        v_train_label,
        learning_rate=0.3,
        epochs=5,
        batch_size=32,
    )
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
    binary_class_results = np.where(yhat > 0.5, 1, 0)
    # get the accuracy score of classification
    logging.info(f"acc: {accuracy_score(y, binary_class_results)}")
    # get the report of classification
    print("classification report:")
    print(classification_report(y, binary_class_results))

Algorithm Details
++++++++++++++++++

the main process is as follows:

    Step 1: Initialize

    - X.rows must be greater than X.cols, otherwise: 1. model will not converge; 2. Y may leak.
    - Y must be held by Bob
    - Initialize w1 / w2, which are the weights corresponding to the features held by Alice / Bob.
    - Use Bob's pk to encrypt w1 -> hw1, and the ciphertext hw1 is stored in Alice.
      Use Alice's pk to encrypt w2 -> hw2, and the ciphertext hw2 is stored in Bob.

    Step 2: Using mini-batch gradient descent, repeat the following steps until the target number of iterations is reached

    - Alice / Bob read x1 / x2, y for current batch as plaintext.
    - Use Bob's pk to encrypt x1 -> hx1, and the ciphertext hx1 is stored in Alice.
      Use Alice's pk to encrypt x2 -> hx2, and the ciphertext hx2 is stored in Bob.
    - Bob infeed y into Secret Sharing <y>
    - Alice locally computes partial predictions hp1 = hx1 * hw1 in homomorphic encryption,
      Bob locally computes partial predictions hp2 = hx2 * hw2 in homomorphic encryption.
    - Convert homomorphic encrypted predictions to secret sharing by H2S operations: H2S(hp1) -> <p1> , H2S(hp2) -> <p2>
    - Calculate <error>=Sigmoid(<p1> + <p2>) - <y> in secret sharing,
      the Sigmoid function approximates using y = 0.5 + 0.125 * x
    - Use Bob's pk to reduce secret sharing to homomorphic encrypted S2H(<error>) -> he1, and the ciphertext he1 is stored in Alice.
      Use Alice's pk to reduce secret sharing to homomorphic encrypted S2H(<error>) -> he2, and the ciphertext he2 is stored in Bob.
    - Alice locally computes hw1 = hw1 - he1 * hx1 * learning_rate in homomorphic encryption,
      Bob locally computes hw2 = hw2 - he2 * hx2 * learning_rate in homomorphic encryption.

    Step 3: Output

    - Convert hw1, hw2 to secret sharing using H2S operation: H2S(hw1) -> <w1> , H2S(hw2) -> <w2>
    - <w> = concatenate(<w1>, <w2>)


Security Analysis
++++++++++++++++++

First, analyze the data interaction in the calculation process to see if there is plaintext information leakage.
There are two types of data interaction in the calculation process:

- -> Marked HE encryption and decryption process and H2S/S2H encryption state conversion:

  + The security of the HE encryption and decryption process completely depends on the algorithm itself.
  + When H2S creates a secret sharing, it will first mark the random number in the ciphertext and then decrypt it,
    without leaking the plaintext information.
  + S2H will first encrypt one party's shard, and then reduce other shards on the ciphertext,
    without leaking plaintext information.

- The interaction in the secret sharing and the computing. The security of these processes depends on the mpc protocol used,
  Taking the default ABY3 protocol as an example, in the case of no collusion between SPU nodes,
  it can be guaranteed that no plaintext information can be returned by analyzing the data exchanged between nodes.

The final output result <w> is stored in the Secret Sharing state, and any w-related information cannot be reversed before reveal <w>.
