Split Learning
==============

What is Split Learning
----------------------

The core idea of split learning is to split the network structure. Each device (silo) retains only a part of the network structure, and the sub-network structure of all devices is combined together to form a complete network model.
In the training process, different devices (silos) only perform forward or reverse calculation on the local network structure, and transfer the calculation results to the next device. Multiple devices complete the training through joint model until convergence.

A typical example of split learning:

.. image:: ../../../tutorial/resources/split_learning_tutorial.png

Alice holds its own data and base model.
Bob holds its own data, base model and fuse model.

1. Alice uses its data to get ``hidden0`` through its base model and send it to Bob.
2. Bob gets ``hidden1`` with its data through its base model.
3. ``hidden_0`` and ``hidden_1`` are input to the ``Agg Layer`` for aggregation, and the aggregated hidden_merge is the output.
4. Bob input hidden_merge to model_fuse, get the gradient with label and send it back.
5. The gradient is split into two parts g_0, g_1 through ``AggLayer``, which are sent to Alice and Bob respectively.
6. Then Alice and Bob update their local base net with g_0 or g_1.


Split Learning Model
--------------------

SecretFlow provides :py:class:`~secretflow_fl.ml.nn.SLModel` to define a split learning model.
You can check the tutorial to have a try.

Tutorial
~~~~~~~~

- :doc:`/tutorial/Split_Learning_for_bank_marketing`
- :doc:`/tutorial/split_learning_gnn`

