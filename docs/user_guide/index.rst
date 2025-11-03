.. _user_guide:

User Guide
============

Preprocessing
-------------

SecretFlow provides several common utility functions/classes to change raw
features into a representation for more suitable for the downstream pipeline.

.. grid:: 2
    :gutter: 2

    .. grid-item-card:: DataFrame
      :link: preprocessing/DataFrame
      :link-type: doc

        Secretflow provides federated data encapsulation in the form of DataFrame.
        DataFrame is composed of data blocks of multiple parties and supports horizontal or vertical partitioned data.

    .. grid-item-card:: WeightOfEnvidenceEncoding (WOE)
      :link:  preprocessing/WeightOfEvidenceEncoding
      :link-type: doc

        Secretflow provides WOE Encoding for vertical datasets.


Private Set Intersection(PSI)
-----------------------------

SecretFlow SPU now supports ECDH-PSI, KKRT16-PSI, and BC22-PCG-PSI.
Please check :doc:`/user_guide/psi` for details.

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

.. toctree::
   :hidden:
   :maxdepth: 2

   preprocessing/index
   psi
   mpc_ml/index
   federated_learning/index

