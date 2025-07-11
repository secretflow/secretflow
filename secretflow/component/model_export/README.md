# Purpose

The model trained by Secretflow requires an online prediction service to be loaded. Online services need to have features such as low latency and high availability. If using prediction component of secretflow such as ` ss_sgd_predict` , the efficiency will be relatively low, and it needs to pass the [component spec](https://github.com/secretflow/spec) interface, which is not convenient to use. [Secretflow-Serving](https://www.secretflow.org.cn/en/docs/serving/) is a framework for online prediction services. It is an online service written in C++. Users can make predictions through [http requests](https://www.secretflow.org.cn/en/docs/serving/reference/api#predictionservice). Compared with Secretflow's prediction component, its performance is much higher and it supports high concurrency. That is to say, the model trained by Secretflow needs to be loaded by Secretflow-Serving. Not only the model, but also the data sources for training and prediction need to be consistent. If Secretflow has performed some feature engineering on the data source before training, such as `onehot`. Since the data sources are consistent, Serving needs to perform the same operation when it receives the predicted data source input and make corresponding preprocessing. 
So `model_export` is a bridge between Secretflow and Serving, which is used to export the preprocessing procedure of the data source and the model trained by Secretflow to Serving. You can also refer to [here](https://www.secretflow.org.cn/en/docs/serving/topics/algorithm/intro) to learn how serving completes the calculations of common models.

# Implementation details

## Model calculation structure


Secretflow's model saves the respective parameters of each party involved in the calculation, exchanges intermediate results during prediction, and finally produces prediction results. The same is true for the [Serving model](https://www.secretflow.org.cn/en/docs/serving/reference/model). Each party has its own parameters. These parameters and calculations are abstracted into a [Graph](https://www.secretflow.org.cn/en/docs/serving/reference/model#graphdef). The Graph contains many [Executions](https://www.secretflow.org.cn/en/docs/serving/reference/model#executiondef) that each party can calculate independently. Network interaction is usually required between each Execution. Within each Execution is computing unit [Nodes](https://www.secretflow.org.cn/en/docs/serving/reference/model#nodedef). The Node executes the basic operator defined by Serving, which is [OP](https://www.secretflow.org.cn/en/docs/serving/reference/model#opdef).
The Execution contained in the current Graph is in the form of a pipeline, that is, the output of the previous one is the input of the next one. The first Execution is executed by each party by default, because theoretically each party needs to input data.
[Secretflow-serving-lib](https://pypi.org/project/secretflow-serving-lib/) also exports the python format of these Message definitions for Secretflow to use.
For details, please refer to the [Secretflow-Serving documentation](https://www.secretflow.org.cn/en/docs/serving).

## Serving operator


The basic operators supported by Serving can be obtained through the [interface](https://github.com/secretflow/serving/blob/main/secretflow_serving_lib/api.py) provided by [secretflow-serving-lib](https://github.com/secretflow/serving/tree/main/secretflow_serving_lib). Each OP has the parameters it needs, You can view the parameters of all operators [here](https://www.secretflow.org.cn/en/docs/serving/topics/graph/operator_list). The protobuf definitions that the definitions of these parameters follow can be found [here](https://github.com/secretflow/serving/blob/main/secretflow_serving/protos/op.proto).

## preprocessing


Secretflow supports many preprocessing components, such as `vert_woe_binning`. These preprocessing processes also need to be passed to Serving, and Serving needs to replay this process. The definition of these processes is abstracted into the addition, deletion, modification and query of data table columns.
After the data table enters Secretflow, all modifications need to use the operations in the [compute package](https://github.com/secretflow/secretflow/tree/main/secretflow/compute). All operations will record [Trace](https://www.secretflow.org.cn/en/docs/serving/reference/model#computetrace) information, which will form a dependency relationship between the input and output of the operation and form a tree structure. By topologically sorting these Traces based on dependencies, the operations can be replayed in the sorted order. These Trace information will be included in the `trace_content` field of [ARROW_PROCESSING](https://www.secretflow.org.cn/en/docs/serving/topics/graph/operator_list#arrow-processing) operator of Serving.

## model file


The calculation process of each party is described by [Graph](https://www.secretflow.org.cn/en/docs/serving/topics/graph/intro_to_graph). The Graph of each party will be placed in the [ModelBundle](https://www.secretflow.org.cn/en/docs/serving/reference/model#modelbundle) structure, which can be exported as Json or Pb serialized string and saved to a file. At the same time, a descriptive file describing the current model package is also required. The format of this file is fixed as Json and the format is [ModelManifest](https://www.secretflow.org.cn/en/docs/serving/reference/model#modelmanifest), which contains the relative path and format information of the ModelBundle. These two files will be packaged into tar files in gzip compression format.
