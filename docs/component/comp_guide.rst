Guide of SecretFlow Components
==============================

Get the Component List
----------------------

Python API
^^^^^^^^^^
You could check SecretFlow Component List by

.. code-block:: python

    from secretflow.component.entry import COMP_LIST

**COMP_LIST** is a CompListDef instance.


CLI
^^^
Check Current SecretFlow Version
++++++++++++++++++++++++++++++++

.. code-block:: sh

    $ secretflow -v
    SecretFlow version 0.8.3b1.

List All Components
++++++++++++++++++++

.. code-block:: sh

    $ secretflow component ls
    DOMAIN                                   NAME                                     VERSION
    ---------------------------------------------------------------------------------------------------------
    feature                                  vert_woe_binning                         0.0.1
    feature                                  vert_woe_substitution                    0.0.1
    ml.eval                                  biclassification_eval                    0.0.1
    ml.eval                                  prediction_bias_eval                     0.0.1
    ml.eval                                  ss_pvalue                                0.0.1
    ml.predict                               sgb_predict                              0.0.1
    ml.predict                               ss_sgd_predict                           0.0.1
    ml.predict                               ss_xgb_predict                           0.0.1
    ml.train                                 sgb_train                                0.0.1
    ml.train                                 ss_sgd_train                             0.0.1
    ml.train                                 ss_xgb_train                             0.0.1
    preprocessing                            feature_filter                           0.0.1
    preprocessing                            psi                                      0.0.1
    preprocessing                            train_test_split                         0.0.1
    stats                                    ss_pearsonr                              0.0.1
    stats                                    ss_vif                                   0.0.1
    stats                                    table_statistics                         0.0.1

Get Definition of Component(s)
++++++++++++++++++++++++++++++

You must specify a component with the following format: **domain/name:version**.

.. code-block:: sh

    $ secretflow component inspect preprocessing/train_test_split:0.0.1
    You are inspecting definition of component with id [preprocessing/train_test_split:0.0.1].
    ---------------------------------------------------------------------------------------------------------
    {
    "domain": "preprocessing",
    "name": "train_test_split",
    "desc": "Split datasets into random train and test subsets. Please check: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html",
    "version": "0.0.1",
    "attrs": [
        {
        "name": "train_size",
        "desc": "Proportion of the dataset to include in the train subset.",
        "type": "AT_FLOAT",
        "atomic": {
            "isOptional": true,
            "defaultValue": {
            "f": 0.75
            },
            "hasLowerBound": true,
            "lowerBound": {},
            "lowerBoundInclusive": true,
            "hasUpperBound": true,
            "upperBound": {
            "f": 1.0
            },
            "upperBoundInclusive": true
        }
        },
        {
        "name": "test_size",
        "desc": "Proportion of the dataset to include in the test subset.",
        "type": "AT_FLOAT",
        "atomic": {
            "isOptional": true,
            "defaultValue": {
            "f": 0.25
            },
            "hasLowerBound": true,
            "lowerBound": {},
            "lowerBoundInclusive": true,
            "hasUpperBound": true,
            "upperBound": {
            "f": 1.0
            },
            "upperBoundInclusive": true
        }
        },
        {
        "name": "random_state",
        "desc": "Specify the random seed of the shuffling.",
        "type": "AT_INT",
        "atomic": {
            "isOptional": true,
            "defaultValue": {
            "i64": "1024"
            },
            "hasLowerBound": true,
            "lowerBound": {}
        }
        },
        {
        "name": "shuffle",
        "desc": "Whether to shuffle the data before splitting.",
        "type": "AT_BOOL",
        "atomic": {
            "isOptional": true,
            "defaultValue": {
            "b": true
            }
        }
        }
    ],
    "inputs": [
        {
        "name": "input_data",
        "desc": "Input dataset.",
        "types": [
            "sf.table.vertical_table"
        ]
        }
    ],
    "outputs": [
        {
        "name": "train",
        "desc": "Output train dataset.",
        "types": [
            "sf.table.vertical_table"
        ]
        },
        {
        "name": "test",
        "desc": "Output test dataset.",
        "types": [
            "sf.table.vertical_table"
        ]
        }
    ]
    }


You could inspect all components at once by

.. code-block:: sh

    $ secretflow component inspect -a
    ...

You may save the list to file by:

.. code-block:: sh

    $ secretflow component inspect -a -f output.json
    You are inspecting the compelete comp list.
    ---------------------------------------------------------------------------------------------------------
    Saved to output.json.


Evaluate a Node
---------------

You should use **secretflow.component.entry.comp_eval** to evaluate a node.

The following code demonstrate how to use this API and could not be run directly.

Python API
^^^^^^^^^^

.. code-block:: python

    import json

    from secretflow.component.entry import comp_eval
    from secretflow.protos.component.cluster_pb2 import (
        SFClusterConfig,
        SFClusterDesc,
        StorageConfig,
    )
    from secretflow.protos.component.comp_pb2 import Attribute
    from secretflow.protos.component.data_pb2 import DistData, TableSchema, VerticalTable
    from secretflow.protos.component.evaluation_pb2 import NodeEvalParam

    desc = SFClusterDesc(
        parties=["alice", "bob"],
        devices=[
            SFClusterDesc.DeviceDesc(
                name="spu",
                type="spu",
                parties=["alice", "bob"],
                config=json.dumps(
                    {
                        "runtime_config": {"protocol": "REF2K", "field": "FM64"},
                        "link_desc": {
                            "connect_retry_times": 60,
                            "connect_retry_interval_ms": 1000,
                            "brpc_channel_protocol": "http",
                            "brpc_channel_connection_type": "pooled",
                            "recv_timeout_ms": 1200 * 1000,
                            "http_timeout_ms": 1200 * 1000,
                        },
                    }
                ),
            ),
            SFClusterDesc.DeviceDesc(
                name="heu",
                type="heu",
                parties=[],
                config=json.dumps(
                    {
                        "mode": "PHEU",
                        "schema": "paillier",
                        "key_size": 2048,
                    }
                ),
            ),
        ],
    )

    sf_cluster_config = SFClusterConfig(
        desc=desc,
        public_config=SFClusterConfig.PublicConfig(
            rayfed_config=SFClusterConfig.RayFedConfig(
                parties=["alice", "bob", "carol", "davy"],
                addresses=[
                    "127.0.0.1:61041",
                    "127.0.0.1:61042",
                ],
            ),
            spu_configs=[
                SFClusterConfig.SPUConfig(
                    name="spu",
                    parties=["alice", "bob"],
                    addresses=[
                        "127.0.0.1:61045",
                        "127.0.0.1:61046",
                    ],
                )
            ],
        ),
        private_config=SFClusterConfig.PrivateConfig(
            self_party="self_party",
            ray_head_addr="local",  # local means setup a Ray cluster instead connecting to an existed one.
            storage_config=StorageConfig(
                type="local_fs",
                local_fs=StorageConfig.LocalFSConfig(wd="storage_path"),
            ),
        ),
    )


    sf_node_eval_param = NodeEvalParam(
        domain="preprocessing",
        name="train_test_split",
        version="0.0.1",
        attr_paths=["train_size", "test_size", "random_state", "shuffle"],
        attrs=[
            Attribute(f=0.75),
            Attribute(f=0.25),
            Attribute(i64=1234),
            Attribute(b=False),
        ],
        inputs=[
            DistData(
                name="input_data",
                type=str("sf.table.vertical_table"),
                data_refs=[
                    DistData.DataRef(uri="bob_input_path", party="bob", format="csv"),
                    DistData.DataRef(uri="alice_input_path", party="alice", format="csv"),
                ],
            )
        ],
        output_uris=[
            "train_output_path",
            "test_output_path",
        ],
    )

    meta = VerticalTable(
        schemas=[
            TableSchema(
                id_types=["str"],
                ids=["id2"],
                feature_types=["f32", "str", "f32"],
                features=["b4", "b5", "b6"],
            ),
            TableSchema(
                id_types=["str"],
                ids=["id1"],
                feature_types=["str", "str", "f32"],
                features=["a1", "a2", "a3"],
                label_types=["f32"],
                labels=["y"],
            ),
        ],
    )

    sf_node_eval_param.inputs[0].meta.Pack(meta)

    res = comp_eval(sf_node_eval_param, sf_cluster_config)


CLI
^^^

.. code-block:: sh

    $ secretflow component run --log_file={log_file} --result_file={result_file_path} --eval_param={encoded_eval_param} --cluster={encoded_cluster_def}


- log_file: log file path.
- result_file: result file path.
- eval_param: base64-encoded NodeEvalParam prototext.
- cluster: base64-encoded SFClusterConfig prototext.

Create a Component
------------------

Python API
^^^^^^^^^^

If you want to create a new component in SecretFlow, you may check one of simplest component:
`secretflow/component/preprocessing/train_test_split.py <https://github.com/secretflow/secretflow/blob/main/secretflow/component/preprocessing/train_test_split.py>`_

The brief steps to build a SecretFlow Component are:

1. Create a new file under **secretflow/component/** .

2. Create a Component class with **secretflow.component.component.Component**:

.. code-block:: python

   from secretflow.component.component import Component

   train_test_split_comp = Component(
       "train_test_split",
       domain="preprocessing",
       version="0.0.1",
       desc="""Split datasets into random train and test subsets.
       Please check: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
       """,
   )

3. Declare attributes and IO.

.. code-block:: python

   from secretflow.component.component import IoType
   from secretflow.component.data_utils import DistDataType

   train_test_split_comp.float_attr(
       name="train_size",
       desc="Proportion of the dataset to include in the train subset.",
       is_list=False,
       is_optional=True,
       default_value=0.75,
       allowed_values=None,
       lower_bound=0.0,
       upper_bound=1.0,
       lower_bound_inclusive=True,
       upper_bound_inclusive=True,
   )
   train_test_split_comp.float_attr(
       name="test_size",
       desc="Proportion of the dataset to include in the test subset.",
       is_list=False,
       is_optional=True,
       default_value=0.25,
       allowed_values=None,
       lower_bound=0.0,
       upper_bound=1.0,
       lower_bound_inclusive=True,
       upper_bound_inclusive=True,
   )
   train_test_split_comp.int_attr(
       name="random_state",
       desc="Specify the random seed of the shuffling.",
       is_list=False,
       is_optional=True,
       default_value=1234,
   )
   train_test_split_comp.bool_attr(
       name="shuffle",
       desc="Whether to shuffle the data before splitting.",
       is_list=False,
       is_optional=True,
       default_value=True,
   )
   train_test_split_comp.io(
       io_type=IoType.INPUT,
       name="input_data",
       desc="Input dataset.",
       types=[DistDataType.VERTICAL_TABLE],
       col_params=None,
   )
   train_test_split_comp.io(
       io_type=IoType.OUTPUT,
       name="train",
       desc="Output train dataset.",
       types=[DistDataType.VERTICAL_TABLE],
       col_params=None,
   )
   train_test_split_comp.io(
       io_type=IoType.OUTPUT,
       name="test",
       desc="Output test dataset.",
       types=[DistDataType.VERTICAL_TABLE],
       col_params=None,
   )

4. Declare evaluation function.

.. code-block:: python

   from secretflow.protos.component.data_pb2 import DistData

   # Signature of eval_fn must be
   #  func(*, ctx, attr_0, attr_1, ..., input_0, input_1, ..., output_0, output_1, ...) -> typing.Dict[str, DistData]
   # All the arguments are keyword-only, so orders don't matter.
   @train_test_split_comp.eval_fn
   def train_test_split_eval_fn(
       *, ctx, train_size, test_size, random_state, shuffle, input_data, train, test
   ):
       # Please check more examples to learn component utils.
       # ctx includes some parsed cluster def and other useful meta.

       # The output of eval_fn is a map of DistDatas of which keys are output names.
       return {"train": DistData(), "test": DistData()}


5. Put your new component in ALL_COMPONENTS of `secretflow.component.entry <https://github.com/secretflow/secretflow/blob/main/secretflow/component/entry.py>`_ .