.. _components:

Components
============

Preprocessing
-------------

SecretFlow provides serveral common utility functions/classes to change raw
features into a representation for more suitable for the downstream pipeline.

.. grid:: 2
    :gutter: 2

    .. grid-item-card:: DataFrame
      :link: preprocessing/DataFrame
      :link-type: doc

        Secretflow provides federated data encapsulation in the form of DataFrame.
        DataFrame is composed of data blocks of multiple parties and supports horizontal or vertical partitioned data.


Private Set Intersection(PSI)
-----------------------------

SecretFlow SPU now supports ECDH-PSI, KKRT16-PSI, and BC22-PCG-PSI. 
Please check :ref:`/components/psi.rst` for details. 

.. grid:: 2
    :gutter: 2

    .. grid-item-card:: Private Set Intersection(PSI)
      :link: psi
      :link-type: doc

        Private set intersection is a secure multiparty computation cryptographic technique that 
        allows two parties holding sets to compare encrypted versions of these sets in order to
        compute the intersection.


MPC Machine Learning
--------------------

SecretFlow provides a variety of MPC modeling capabilities
through the MPC security protocol and HE homomorphic encryption.

.. grid:: 2
    :gutter: 2

    .. grid-item-card:: Linear Models
      :link: mpc_ml/linear_model
      :link-type: doc

        a set of methods intended for regression in which the mean of target value is expected to be
        a linear combination of the features (or a map of linear combinations).

    .. grid-item-card:: Decision trees
      :link: mpc_ml/decision_tree
      :link-type: doc

        Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression.

    .. grid-item-card:: Feature Engineering
      :link: mpc_ml/feature_eng
      :link-type: doc

        Feature Engineering includes **Pearson product-moment correlation coefficient**, **Variance Inflation Factor (VIF)** and **Hypothesis Testing for linear Regression Coefficients**.

Federated Learning
------------------

Federated learning is a machine learning technique that trains an algorithm across multiple decentralized edge devices or servers holding local data samples, without exchanging them.


.. grid:: 2
    :gutter: 2

    .. grid-item-card:: Horizontal Federated Learning
      :link: federated_learning/horizontal_federated_learning/index
      :link-type: doc

        For cases that multi participants share the same feature space but differ in sample ID.

    .. grid-item-card:: Vertical Federated Learning
      :link: federated_learning/vertical_federated_learning
      :link-type: doc

        For cases that multi participants share the same sample ID space but differ in feature space.

    .. grid-item-card:: Mix Federated Learning
      :link: federated_learning/mix_federated_learning
      :link-type: doc

        For cases that parts of participants share the same sample ID space but differ in feature space, 
        where others share the same feature space but differ in sample ID.

.. toctree::
   :hidden:
   :maxdepth: 2

   preprocessing/index
   psi
   mpc_ml/index
   federated_learning/index

