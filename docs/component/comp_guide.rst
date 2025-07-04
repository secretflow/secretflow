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

e.g. Let's check the component definition of PSI.

.. code-block:: sh

    $ secretflow component inspect preprocessing/psi:0.0.1
    You are inspecting definition of component with id [preprocessing/psi:0.0.1].
    ---------------------------------------------------------------------------------------------------------
    {
    "domain": "preprocessing",
    "name": "psi",
    "desc": "PSI between two parties.",
    "version": "0.0.1",
    "attrs": [
        {
        "name": "protocol",
        "desc": "PSI protocol.",
        "type": "AT_STRING",
        "atomic": {
            "isOptional": true,
            "defaultValue": {
            "s": "PROTOCOL_ECDH"
            },
            "allowedValues": {
            "ss": [
                "PROTOCOL_ECDH",
                "PROTOCOL_KKRT",
                "PROTOCOL_RR22"
            ]
            }
        }
        },
        {
        "name": "bucket_size",
        "desc": "Specify the hash bucket size used in PSI. Larger values consume more memory.",
        "type": "AT_INT",
        "atomic": {
            "isOptional": true,
            "defaultValue": {
            "i64": "1048576"
            },
            "lowerBoundEnabled": true,
            "lowerBound": {}
        }
        },
        {
        "name": "ecdh_curve_type",
        "desc": "Curve type for ECDH PSI.",
        "type": "AT_STRING",
        "atomic": {
            "isOptional": true,
            "defaultValue": {
            "s": "CURVE_FOURQ"
            },
            "allowedValues": {
            "ss": [
                "CURVE_25519",
                "CURVE_FOURQ",
                "CURVE_SM2",
                "CURVE_SECP256K1"
            ]
            }
        }
        }
    ],
    "inputs": [
        {
        "name": "receiver_input",
        "desc": "Individual table for receiver",
        "types": [
            "sf.table.individual"
        ],
        "attrs": [
            {
            "name": "key",
            "desc": "Column(s) used to join. If not provided, ids of the dataset will be used."
            }
        ]
        },
        {
        "name": "sender_input",
        "desc": "Individual table for sender",
        "types": [
            "sf.table.individual"
        ],
        "attrs": [
            {
            "name": "key",
            "desc": "Column(s) used to join. If not provided, ids of the dataset will be used."
            }
        ]
        }
    ],
    "outputs": [
        {
        "name": "psi_output",
        "desc": "Output vertical table",
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

Python API
^^^^^^^^^^

In the following examples, we would demonstrate how to evaluate a node with Python API.

We are going to test PSI component with tiny datasets.

1. Save the following bash script as *generate_csv.sh*

.. code-block:: bash

    #!/bin/bash

    set -e
    show_help() {
        echo "Usage: bash generate_csv.sh -c {col_name} -p {file_name}"
        echo "  -c"
        echo "          the column name of id."
        echo "  -p"
        echo "          the path of output csv."
    }
    if [[ "$#" -lt 1 ]]; then
        show_help
        exit
    fi

    while getopts ":c:p:" OPTION; do
        case $OPTION in
        c)
            COL_NAME=$OPTARG
            ;;
        p)
            FILE_PATH=$OPTARG
            ;;
        *)
            echo "Incorrect options provided"
            exit 1
            ;;
        esac
    done


    # header
    echo $COL_NAME > $FILE_PATH

    # generate 800 random int
    for ((i=0; i<800; i++))
    do
    # from 0 to 1000
    id=$(shuf -i 0-1000 -n 1)

    # check duplicates
    while grep -q "^$id$" $FILE_PATH
    do
        id=$(shuf -i 0-1000 -n 1)
    done

    # write
    echo "$id" >> $FILE_PATH
    done

    echo "Generated csv file is $FILE_PATH."


Then generate input for two parties.

.. code-block:: bash

    mkdir -p /tmp/alice
    sh generate_csv.sh -c id1 -p /tmp/alice/input.csv

    mkdir -p /tmp/bob
    sh generate_csv.sh -c id2 -p /tmp/bob/input.csv


2. Save the following Python code as *psi_demo.py*

.. code-block:: python

    import json

    from secretflow.component.core import comp_eval
    from secretflow.spec.extend.cluster_pb2 import (
        SFClusterConfig,
        SFClusterDesc,
    )
    from secretflow_spec.v1.component_pb2 import Attribute
    from secretflow_spec.v1.data_pb2 import (
        DistData,
        TableSchema,
        IndividualTable,
        StorageConfig,
    )
    from secretflow_spec.v1.evaluation_pb2 import NodeEvalParam
    import click


    @click.command()
    @click.argument("party", type=str)
    def run(party: str):
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
                ray_fed_config=SFClusterConfig.RayFedConfig(
                    parties=["alice", "bob"],
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
                self_party=party,
                ray_head_addr="local",  # local means setup a Ray cluster instead connecting to an existed one.
            ),
        )

        # check https://www.secretflow.org.cn/docs/spec/latest/zh-Hans/intro#nodeevalparam for details.
        sf_node_eval_param = NodeEvalParam(
            domain="preprocessing",
            name="psi",
            version="0.0.1",
            attr_paths=[
                "protocol",
                "sort",
                "bucket_size",
                "ecdh_curve_type",
                "input/receiver_input/key",
                "input/sender_input/key",
            ],
            attrs=[
                Attribute(s="PROTOCOL_ECDH"),
                Attribute(b=True),
                Attribute(i64=1048576),
                Attribute(s="CURVE_FOURQ"),
                Attribute(ss=["id1"]),
                Attribute(ss=["id2"]),
            ],
            inputs=[
                DistData(
                    name="receiver_input",
                    type="sf.table.individual",
                    data_refs=[
                        DistData.DataRef(uri="input.csv", party="alice", format="csv"),
                    ],
                ),
                DistData(
                    name="sender_input",
                    type="sf.table.individual",
                    data_refs=[
                        DistData.DataRef(uri="input.csv", party="bob", format="csv"),
                    ],
                ),
            ],
            output_uris=[
                "output.csv",
            ],
        )

        sf_node_eval_param.inputs[0].meta.Pack(
            IndividualTable(
                schema=TableSchema(
                    id_types=["str"],
                    ids=["id1"],
                ),
                line_count=-1,
            ),
        )

        sf_node_eval_param.inputs[1].meta.Pack(
            IndividualTable(
                schema=TableSchema(
                    id_types=["str"],
                    ids=["id2"],
                ),
                line_count=-1,
            ),
        )

        storage_config = StorageConfig(
            type="local_fs",
            local_fs=StorageConfig.LocalFSConfig(wd=f"/tmp/{party}"),
        )

        res = comp_eval(sf_node_eval_param, storage_config, sf_cluster_config)

        print(f'Node eval res is \n{res}')


    if __name__ == "__main__":
        run()


3. In two separate terminals, run

.. code-block:: python

    $ python psi_demo.py alice

.. code-block:: python

    $ python psi_demo.py bob

You should see the following output at both terminals:

.. code-block:: python

   Node eval res is
   outputs {
     name: "output.csv"
     type: "sf.table.vertical_table"
     system_info {
     }
     meta {
       type_url: "type.googleapis.com/secretflow_spec.v1.VerticalTable"
       value: "\n\n\n\003id1\"\003str\n\n\n\003id2\"\003str\020\211\005"
     }
     data_refs {
       uri: "output.csv"
       party: "alice"
       format: "csv"
     }
     data_refs {
       uri: "output.csv"
       party: "bob"
       format: "csv"
     }
   }

4. Check result at */tmp/alice/output.csv* and */tmp/bob/output.csv*. The content of two files should be same except the header.

CLI
^^^

You could also use SecretFlow CLI to evaluate a node.

.. code-block:: sh

    $ secretflow component run --log_file={log_file} --result_file={result_file_path} --eval_param={encoded_eval_param} --storage={encoded_storage_config} --cluster={encoded_cluster_def}


- log_file: log file path.
- result_file: result file path.
- eval_param: base64-encoded NodeEvalParam prototext.
- storage: base64-encoded StorageConfig prototext.
- cluster: base64-encoded SFClusterConfig prototext.

Since you need to encode prototext to use CLI, we don't expect you to use SecretFlow CLI for node evaluation.

Create a Component
------------------

Python API
^^^^^^^^^^

If you want to create a new component in SecretFlow, you may check one of simplest component:
`secretflow/component/preprocessing/data_prep/train_test_split.py <https://github.com/secretflow/secretflow/blob/main/secretflow/component/preprocessing/data_prep/train_test_split.py>`_

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
   from secretflow.component.core import DistDataType

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

   from secretflow_spec.v1.data_pb2 import DistData

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
