



SecretFlow Component List
=========================


Version: 1.0

First-party SecretFlow components.
## data_filter

### condition_filter


Component version: 1.0.0

Filter the table based on a single column's values and condition.
Warning: the party responsible for condition filtering will directly send the sample distribution to other participants.
Malicious participants can obtain the distribution of characteristics by repeatedly calling with different filtering values.
Audit the usage of this component carefully.
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|comparator|Comparator to use for comparison. Must be one of '==','<','<=','>','>=','IN','NOTNULL'|String|Y|Allowed: ['==', '<', '<=', '>', '>=', 'IN', 'NOTNULL'].|
|bound_value|Input a value for comparison; if the comparison condition is IN, you can input multiple values separated by ','; if the comparison condition is NOTNULL, the input is not needed.|String|N|Default: .|
|float_epsilon|Epsilon value for floating point comparison. WARNING: due to floating point representation in computers, set this number slightly larger if you want filter out the values exactly at desired boundary. for example, abs(1.001 - 1.002) is slightly larger than 0.001, and therefore may not be filter out using == and epsilson = 0.001|Float|N|Default: 0.0.Range: [0.0, $\infty$).|

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_ds|Input vertical table.|['sf.table.vertical']|Pleae fill in extra table attributes.|
|input/input_ds/feature|Feature to operate on.|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. Min column number to select(inclusive): 1. Max column number to select(inclusive): 1. |

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_ds|Output vertical table that satisfies the condition.|['sf.table.vertical']||
|output_ds_else|Output vertical table that does not satisfies the condition.|['sf.table.vertical']||

### expr_condition_filter


Component version: 1.0.0

Only row-level filtering is supported, column processing is not available;
the custom expression must comply with SQLite syntax standards
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|expr|The custom expression must comply with SQLite syntax standards|String|Y||

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_ds|Input vertical or individual table|['sf.table.individual', 'sf.table.vertical']||

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_ds|Output table that satisfies the condition|['sf.table.individual', 'sf.table.vertical']||
|output_ds_else|Output table that does not satisfies the condition|['sf.table.individual', 'sf.table.vertical']||

### feature_filter


Component version: 1.0.0

Drop features from the dataset.
#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_ds|Input vertical table.|['sf.table.vertical']|Pleae fill in extra table attributes.|
|input/input_ds/drop_features|Features to drop.|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. Min column number to select(inclusive): 1. |

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_ds|Output vertical table.|['sf.table.vertical']||

### sample


Component version: 1.0.0

Sample data set.
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|sample_algorithm|sample algorithm and parameters|Special type. Union group. You must select one child to fill in.|N/A|This is a special type. This is a union group, you must select one child to fill in (if exists).|
|sample_algorithm/random|Random sample.|Special type. Struct group. You must fill in all children.|N/A|This is a special type. This is a structure group, you must fill in all children.|
|sample_algorithm/random/frac|Proportion of the dataset to sample in the set. The fraction should be larger than 0.|Float|N|Default: 0.8.Range: (0.0, 10000.0).|
|sample_algorithm/random/random_state|Specify the random seed of the shuffling.|Integer|N|Default: 1024.Range: (0, $\infty$).|
|sample_algorithm/random/replacement|If true, sampling with replacement. If false, sampling without replacement.|Boolean|N|Default: False.|
|sample_algorithm/system|system sample.|Special type. Struct group. You must fill in all children.|N/A|This is a special type. This is a structure group, you must fill in all children.|
|sample_algorithm/system/frac|Proportion of the dataset to sample in the set. The fraction should be larger than 0 and less than or equal to 0.5.|Float|N|Default: 0.2.Range: (0.0, 0.5].|
|sample_algorithm/stratify|stratify sample.|Special type. Struct group. You must fill in all children.|N/A|This is a special type. This is a structure group, you must fill in all children.|
|sample_algorithm/stratify/frac|Proportion of the dataset to sample in the set. The fraction should be larger than 0.|Float|N|Default: 0.8.Range: (0.0, 10000.0).|
|sample_algorithm/stratify/random_state|Specify the random seed of the shuffling.|Integer|N|Default: 1024.Range: (0, $\infty$).|
|sample_algorithm/stratify/observe_feature|stratify sample observe feature.|String|Y||
|sample_algorithm/stratify/replacements|If true, sampling with replacement. If false, sampling without replacement.|Boolean List|Y||
|sample_algorithm/stratify/quantiles|stratify sample quantiles|Float List|Y|Min length(inclusive): 1. Max length(inclusive): 1000.|
|sample_algorithm/stratify/weights|stratify sample weights|Float List|N|Default: [].Range: ([], []).|

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_ds|Input vertical table.|['sf.table.vertical', 'sf.table.individual']||

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_ds|Output sampled dataset.|['sf.table.vertical', 'sf.table.individual']||
|report|Output sample report|['sf.report']||

## data_prep

### psi


Component version: 1.0.0

PSI between two parties.
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|protocol|PSI protocol.|Special type. Union group. You must select one child to fill in.|N/A|This is a special type. This is a union group, you must select one child to fill in (if exists).|
|protocol/PROTOCOL_ECDH|ECDH protocol.|Special type. Union group. You must select one child to fill in.|N/A|This is a special type. This is a union group, you must select one child to fill in (if exists).|
|sort_result|If false, output is not promised to be aligned. Warning: disable this option may lead to errors in the following components. DO NOT TURN OFF if you want to append other components.|Boolean|N|Default: True.|
|receiver_parties|Party names of receiver for result, all party will be receivers default; if only one party receive result, the result will be single-party table, hence you can not connect it to component with union table input.|Special type. Specify parties.|Y||
|allow_empty_result|Whether to allow the result to be empty, if allowed, an empty file will be saved, if not, an error will be reported.|Boolean|N|Default: False.|
|join_type|join type, default is inner join.|Special type. Union group. You must select one child to fill in.|N/A|This is a special type. This is a union group, you must select one child to fill in (if exists).|
|join_type/left_join|Left join|Special type. Struct group. You must fill in all children.|N/A|This is a special type. This is a structure group, you must fill in all children.|
|join_type/left_join/left_side|Required for left join|Special type. Specify parties.|Y||
|input_ds1_keys_duplicated|Whether key columns have duplicated rows, default is True.|Boolean|N|Default: True.|
|input_ds2_keys_duplicated|Whether key columns have duplicated rows, default is True.|Boolean|N|Default: True.|

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_ds1|Individual table for party 1|['sf.table.individual']|Pleae fill in extra table attributes.|
|input/input_ds1/keys|Column(s) used to join.|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds1. Min column number to select(inclusive): 1. |
|input_ds2|Individual table for party 2|['sf.table.individual']|Pleae fill in extra table attributes.|
|input/input_ds2/keys|Column(s) used to join.|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds2. Min column number to select(inclusive): 1. |

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_ds|Output vertical table|['sf.table.vertical', 'sf.table.individual']||
|report|Output psi report|['sf.report']||

### psi_tp


Component version: 1.0.0

PSI between three parties.
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|ecdh_curve|Curve type for ECDH PSI.|Special type. Union group. You must select one child to fill in.|N/A|This is a special type. This is a union group, you must select one child to fill in (if exists).|

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_ds1|Individual table for party 1|['sf.table.individual']|Pleae fill in extra table attributes.|
|input/input_ds1/keys1|Column(s) used to join.|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds1. Min column number to select(inclusive): 1. |
|input_ds2|Individual table for party 2|['sf.table.individual']|Pleae fill in extra table attributes.|
|input/input_ds2/keys2|Column(s) used to join.|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds2. Min column number to select(inclusive): 1. |
|input_ds3|Individual table for party 3|['sf.table.individual']|Pleae fill in extra table attributes.|
|input/input_ds3/keys3|Column(s) used to join.|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds3. Min column number to select(inclusive): 1. |

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_ds|Output vertical table|['sf.table.vertical']||

### train_test_split


Component version: 1.0.0

Split datasets into random train and test subsets.
- Please check: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|train_size|Proportion of the dataset to include in the train subset. The sum of test_size and train_size should be in the (0, 1] range.|Float|N|Default: 0.75.Range: (0.0, 1.0).|
|test_size|Proportion of the dataset to include in the test subset. The sum of test_size and train_size should be in the (0, 1] range.|Float|N|Default: 0.25.Range: (0.0, 1.0).|
|random_state|Specify the random seed of the shuffling.|Integer|N|Default: 1024.Range: (0, $\infty$).|
|shuffle|Whether to shuffle the data before splitting.|Boolean|N|Default: True.|

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_ds|Input vertical table.|['sf.table.vertical']||

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|train_ds|Output train dataset.|['sf.table.vertical']||
|test_ds|Output test dataset.|['sf.table.vertical']||

### unbalance_psi


Component version: 1.0.0

Unbalance psi with cache.
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|join_type|join type, default is inner join.|Special type. Union group. You must select one child to fill in.|N/A|This is a special type. This is a union group, you must select one child to fill in (if exists).|
|join_type/left_join|Left join|Special type. Struct group. You must fill in all children.|N/A|This is a special type. This is a structure group, you must fill in all children.|
|join_type/left_join/left_side|Required for left join|Special type. Specify parties.|Y||
|allow_empty_result|Whether to allow the result to be empty, if allowed, an empty file will be saved, if not, an error will be reported.|Boolean|N|Default: False.|
|receiver_parties|Party names of receiver for result, all party will be receivers default; if only one party receive result, the result will be single-party table, hence you can not connect it to component with union table input.|Special type. Specify parties.|Y||

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|client_ds|Client dataset.|['sf.table.individual']|Pleae fill in extra table attributes.|
|input/client_ds/keys|Keys to be used for psi.|String List(Set value with other Component Attributes)|You need to select some columns of table client_ds. |
|cache|Server cache.|['sf.model.ub_psi.cache']||

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_ds|Output table|['sf.table.individual', 'sf.table.vertical']||

### unbalance_psi_cache


Component version: 1.0.0

Generate cache for unbalance psi on both sides.
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|client|Party of client(party with the smaller dataset).|Special type. Specify parties.|Y||

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_ds|Input vertical table.|['sf.table.individual']|Pleae fill in extra table attributes.|
|input/input_ds/keys|Keys to be used for psi.|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. Min column number to select(inclusive): 1. |

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_cache|Output cache.|['sf.model.ub_psi.cache']||

### union


Component version: 1.0.0

Perform a horizontal merge of two data tables, supporting the individual table or vertical table on the same node.
#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_ds1|The first input table|['sf.table.individual', 'sf.table.vertical']||
|input_ds2|The second input table|['sf.table.individual', 'sf.table.vertical']||

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_ds|Output table|['sf.table.individual', 'sf.table.vertical']||

## io

### data_sink


Component version: 1.0.0

export data to an external data source
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|output_party|output party|Special type. Specify parties.|Y||
|output_uri|output uri, the uri format is datamesh:///{relative_path}?domaindata_id={domaindata_id}&datasource_id={datasource_id}&partition_spec={partition_spec}|String|Y||

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_data|Input dist data|['sf.table.individual', 'sf.table.vertical']||

### data_source


Component version: 1.0.0

import data from an external data source
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|party||Special type. Specify parties.|Y||
|uri|input uri, the uri format is datamesh:///{relative_path}?domaindata_id={domaindata_id}&datasource_id={datasource_id}&partition_spec={partition_spec}|String|Y||
|columns|table column info, json format, for example {"col1": "ID", "col2":"FEATURE", "col3":"LABEL"}|String|N|Default: .|

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_ds|output dataset|['sf.table.individual']||

### identity


Component version: 1.0.0

map any input to output
#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_data|Input data|['sf.model.ss_glm', 'sf.model.sgb', 'sf.model.ss_xgb', 'sf.model.ss_sgd', 'sf.rule.binning', 'sf.read_data']||

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_data|Output data|['sf.model.ss_glm', 'sf.model.sgb', 'sf.model.ss_xgb', 'sf.model.ss_sgd', 'sf.rule.binning', 'sf.read_data']||

### read_data


Component version: 1.0.0

read model or rules from sf cluster
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|generalized_linear_model|Whether to dump the complete generalized linear model. The complete generalized linear model contains link, y_scale, offset_col, and so on.|Boolean|N|Default: False.|

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_data|Input dist data|['sf.rule.binning', 'sf.model.ss_glm', 'sf.model.sgb']||

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_data|Output rules or models in DistData.meta|['sf.read_data']||

### write_data


Component version: 1.0.0

write model or rules back to sf cluster
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|write_data|rule or model protobuf by json format|String|Y||
|write_data_type|which rule or model is writing|String|N|Default: sf.rule.binning.Allowed: ['sf.rule.binning', 'sf.model.ss_glm', 'sf.model.sgb'].|

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_data|Input dist data. Rule reconstructions may need hidden info in original rule for security considerations.|['sf.rule.binning', 'sf.model.ss_glm', 'sf.model.sgb', 'sf.null']||

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_data|Output rules or models in sf cluster format|['sf.rule.binning', 'sf.model.ss_glm', 'sf.model.sgb']||

## ml.eval

### biclassification_eval


Component version: 1.0.0

Statistics evaluation for a bi-classification model on a dataset.
1. summary_report: SummaryReport
2. eq_frequent_bin_report: List[EqBinReport]
3. eq_range_bin_report: List[EqBinReport]
4. head_report: List[PrReport]
reports for fpr = 0.001, 0.005, 0.01, 0.05, 0.1, 0.2
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|bucket_size|Number of buckets.|Integer|N|Default: 10.Range: [1, $\infty$).|
|min_item_cnt_per_bucket|Min item cnt per bucket. If any bucket doesn't meet the requirement, error raises. For security reasons, we require this parameter to be at least 5.|Integer|N|Default: 5.Range: [5, $\infty$).|

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_ds|Input table with prediction and label, usually is a result from a prediction component.|['sf.table.vertical', 'sf.table.individual']|Pleae fill in extra table attributes.|
|input/input_ds/label|The label name to use in the dataset.|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. Min column number to select(inclusive): 1. Max column number to select(inclusive): 1. |
|input/input_ds/prediction|The prediction result column name to use in the dataset.|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. Min column number to select(inclusive): 1. Max column number to select(inclusive): 1. |

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|report|Output report.|['sf.report']||

### prediction_bias_eval


Component version: 1.0.0

Calculate prediction bias, ie. average of predictions - average of labels.
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|bucket_num|Num of bucket.|Integer|N|Default: 10.Range: [1, $\infty$).|
|min_item_cnt_per_bucket|Min item cnt per bucket. If any bucket doesn't meet the requirement, error raises. For security reasons, we require this parameter to be at least 2.|Integer|N|Default: 2.Range: [2, $\infty$).|
|bucket_method|Bucket method.|String|N|Default: equal_width.Allowed: ['equal_width', 'equal_frequency'].|

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_ds|Input table with prediction and label, usually is a result from a prediction component.|['sf.table.vertical', 'sf.table.individual']|Pleae fill in extra table attributes.|
|input/input_ds/label|The label name to use in the dataset.|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. Min column number to select(inclusive): 1. Max column number to select(inclusive): 1. |
|input/input_ds/prediction|The prediction result column name to use in the dataset.|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. Min column number to select(inclusive): 1. Max column number to select(inclusive): 1. |

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|report|Output report.|['sf.report']||

### regression_eval


Component version: 1.0.0

Statistics evaluation for a regression model on a dataset.
Contained Statistics:
R2 Score (r2_score): It is a statistical measure that represents the proportion of the variance in the dependent variable that can be predicted from the independent variables. It ranges from -inf to 1, where a higher value indicates a better fit. (the value can be negative because the
model can be arbitrarily worse). In the general case when the true y is non-constant, a constant model that always predicts the average y
disregarding the input features would get a :math:'R^2' score of 0.0.
Mean Absolute Error (mean_abs_err): It calculates the average absolute difference between the predicted and actual values. It provides a measure of the average magnitude of the errors.
Mean Absolute Percentage Error (mean_abs_percent_err): It calculates the average absolute percentage difference between the predicted and actual values. It measures the average magnitude of the errors in terms of percentages.
Sum of Squared Errors (sum_squared_errors): It calculates the sum of the squared differences between the predicted and actual values. It provides an overall measure of the model's performance.
Mean Squared Error (mean_squared_errors): It calculates the average of the squared differences between the predicted and actual values. It is widely used as a loss function in regression problems.
Root Mean Squared Error (root_mean_squared_errors): It is the square root of the mean squared error. It provides a measure of the average magnitude of the errors in the original scale of the target variable.
Mean of True Values (y_true_mean): It calculates the average of the actual values in the target variable. It can be useful for establishing a baseline for the model's performance.
Mean of Predicted Values (y_pred_mean): It calculates the average of the predicted values. It can be compared with the y_true_mean to get an idea of the model's bias.
Residual Histograms (residual_hists): It represents the distribution of the differences between the predicted and actual values. It helps to understand the spread and pattern of the errors.
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|bucket_size|Number of buckets for residual histogram.|Integer|N|Default: 10.Range: [1, 10000].|

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_ds|Input table with prediction and label, usually is a result from a prediction component.|['sf.table.vertical', 'sf.table.individual']|Pleae fill in extra table attributes.|
|input/input_ds/label|The label name to use in the dataset.|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. Min column number to select(inclusive): 1. Max column number to select(inclusive): 1. |
|input/input_ds/prediction|The prediction result column name to use in the dataset.|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. Min column number to select(inclusive): 1. Max column number to select(inclusive): 1. |

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|report|Output report.|['sf.report']||

### ss_pvalue


Component version: 1.0.0

Calculate P-Value for LR model training on vertical partitioning dataset by using secret sharing.
For large dataset(large than 10w samples & 200 features),
recommend to use [Ring size: 128, Fxp: 40] options for SPU device.
#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_model|Input model.|['sf.model.ss_sgd', 'sf.model.ss_glm']||
|input_ds|Input vertical table.|['sf.table.vertical']||

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|report|Output P-Value report.|['sf.report']||

## ml.predict

### gnb_predict


Component version: 1.0.0

Predict using the gaussian naive bayes model. This component is currently experimental.
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|receiver|Party of receiver.|Special type. Specify parties.|Y||
|pred_name|Column name for predictions.|String|N|Default: pred.|
|save_ids|Whether to save ids columns into output prediction table. If true, input feature_dataset must contain id columns, and receiver party must be id owner.|Boolean|N|Default: False.|
|save_label|Whether or not to save real label columns into output pred file. If true, input feature_dataset must contain label columns and receiver party must be label owner.|Boolean|N|Default: True.|

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_model|model|['sf.model.gnb']||
|input_ds|Input vertical table.|['sf.table.vertical']|Pleae fill in extra table attributes.|
|input/input_ds/saved_features|which features should be saved with prediction result|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. |

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_ds|Output prediction.|['sf.table.individual']||

### gpc_predict


Component version: 1.0.0

Predict using the gaussian process classifier model. This component is currently experimental.
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|receiver|Party of receiver.|Special type. Specify parties.|Y||
|pred_name|Column name for predictions.|String|N|Default: pred.|
|save_ids|Whether to save ids columns into output prediction table. If true, input feature_dataset must contain id columns, and receiver party must be id owner.|Boolean|N|Default: False.|
|save_label|Whether or not to save real label columns into output pred file. If true, input feature_dataset must contain label columns and receiver party must be label owner.|Boolean|N|Default: True.|

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_model|model|['sf.model.gpc']||
|input_ds|Input vertical table.|['sf.table.vertical']|Pleae fill in extra table attributes.|
|input/input_ds/saved_features|which features should be saved with prediction result|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. |

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_ds|Output prediction.|['sf.table.individual']||

### kmeans_predict


Component version: 1.0.0

Predict using the KMeans model. This component is currently experimental.
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|receiver|Party of receiver.|Special type. Specify parties.|Y||
|pred_name|Column name for predictions.|String|N|Default: pred.|
|save_ids|Whether to save ids columns into output prediction table. If true, input feature_dataset must contain id columns, and receiver party must be id owner.|Boolean|N|Default: False.|

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_model|model|['sf.model.kmeans']||
|input_ds|Input vertical table.|['sf.table.vertical']|Pleae fill in extra table attributes.|
|input/input_ds/saved_features|which features should be saved with prediction result|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. |

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_ds|Output prediction.|['sf.table.individual']||

### knn_predict


Component version: 1.0.0

Predict using the K neighbors classifier model. This component is currently experimental.
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|receiver|Party of receiver.|Special type. Specify parties.|Y||
|pred_name|Column name for predictions.|String|N|Default: pred.|
|save_ids|Whether to save ids columns into output prediction table. If true, input feature_dataset must contain id columns, and receiver party must be id owner.|Boolean|N|Default: False.|
|save_label|Whether or not to save real label columns into output pred file. If true, input feature_dataset must contain label columns and receiver party must be label owner.|Boolean|N|Default: True.|

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_model|model|['sf.model.knn']||
|input_ds|Input vertical table.|['sf.table.vertical']|Pleae fill in extra table attributes.|
|input/input_ds/saved_features|which features should be saved with prediction result|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. |

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_ds|Output prediction.|['sf.table.individual']||

### serving_model_inferencer


Component version: 1.1.0

batch predicting online service models in offline
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|receiver|Party of receiver.|Special type. Specify parties.|Y||
|pred_name|Column name for predictions.|String|N|Default: score.|
|input_block_size|block size (Byte) for input data streaming|Integer|N|Default: 65536.|

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|serving_model|Input serving model.|['sf.serving.model']||
|input_ds|Input vertical table or individual table.|['sf.table.vertical', 'sf.table.individual']|Pleae fill in extra table attributes.|
|input/input_ds/saved_columns|which columns should be saved with prediction result|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. |

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_ds|Output prediction.|['sf.table.individual']||

### sgb_predict


Component version: 1.0.0

Predict using SGB model.
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|receiver|Party of receiver.|Special type. Specify parties.|Y||
|pred_name|Name for prediction column|String|N|Default: pred.|
|save_ids|Whether to save ids columns into output prediction table. If true, input feature_dataset must contain id columns, and receiver party must be id owner.|Boolean|N|Default: False.|
|save_label|Whether or not to save real label columns into output pred file. If true, input feature_dataset must contain label columns and receiver party must be label owner.|Boolean|N|Default: True.|

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_model|model|['sf.model.sgb']||
|input_ds|Input vertical table.|['sf.table.vertical']|Pleae fill in extra table attributes.|
|input/input_ds/saved_features|which features should be saved with prediction result|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. |

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_ds|Output prediction.|['sf.table.individual']||

### ss_glm_predict


Component version: 1.1.0

Predict using the SSGLM model.
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|receiver|Party of receiver.|Special type. Specify parties.|Y||
|pred_name|Column name for predictions.|String|N|Default: pred.|
|save_ids|Whether to save ids columns into output prediction table. If true, input feature_dataset must contain id columns, and receiver party must be id owner.|Boolean|N|Default: True.|
|save_label|Whether or not to save real label columns into output pred file. If true, input feature_dataset must contain label columns and receiver party must be label owner.|Boolean|N|Default: False.|

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_model|Input model.|['sf.model.ss_glm']||
|input_ds|Input vertical table.|['sf.table.vertical']|Pleae fill in extra table attributes.|
|input/input_ds/saved_features|which features should be saved with prediction result|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. |

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_ds|Output prediction.|['sf.table.individual']||

### ss_sgd_predict


Component version: 1.0.0

Predict using the SS-SGD model.
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|batch_size|The number of training examples utilized in one iteration.|Integer|N|Default: 1024.Range: (0, $\infty$).|
|receiver|Party of receiver.|Special type. Specify parties.|Y||
|pred_name|Column name for predictions.|String|N|Default: pred.|
|save_ids|Whether to save ids columns into output prediction table. If true, input feature_dataset must contain id columns, and receiver party must be id owner.|Boolean|N|Default: True.|
|save_label|Whether or not to save real label columns into output pred file. If true, input feature_dataset must contain label columns and receiver party must be label owner.|Boolean|N|Default: False.|

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_model|Input model.|['sf.model.ss_sgd']||
|input_ds|Input vertical table.|['sf.table.vertical']|Pleae fill in extra table attributes.|
|input/input_ds/saved_features|which features should be saved with prediction result|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. |

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_ds|Output prediction.|['sf.table.individual']||

### ss_xgb_predict


Component version: 1.0.0

Predict using the SS-XGB model.
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|receiver|Party of receiver.|Special type. Specify parties.|Y||
|pred_name|Column name for predictions.|String|N|Default: pred.|
|save_ids|Whether to save ids columns into output prediction table. If true, input feature_dataset must contain id columns, and receiver party must be id owner.|Boolean|N|Default: False.|
|save_label|Whether or not to save real label columns into output pred file. If true, input feature_dataset must contain label columns and receiver party must be label owner.|Boolean|N|Default: True.|

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_model|model|['sf.model.ss_xgb']||
|input_ds|Input vertical table.|['sf.table.vertical']|Pleae fill in extra table attributes.|
|input/input_ds/saved_features|which features should be saved with prediction result|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. |

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_ds|Output prediction.|['sf.table.individual']||

## ml.train

### gnb_train


Component version: 1.0.0

Provide gaussian naive bayes training. This component is currently experimental.
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|var_smoothing|Portion of the largest variance of all features that is added to variances for calculation stability.|Float|N|Default: 0.0.Range: (0.0, $\infty$).|
|n_classes|The number of classes in the training data, must be preprocessed to 0, 1, 2, ..., n_classes - 1|Integer|N|Default: 2.Range: [2, $\infty$).|

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_ds|Input vertical table.|['sf.table.vertical']|Pleae fill in extra table attributes.|
|input/input_ds/feature_selects|which features should be used for training.|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. Min column number to select(inclusive): 1. |
|input/input_ds/label|Label of train dataset.|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. Min column number to select(inclusive): 1. Max column number to select(inclusive): 1. |

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_model|Output model.|['sf.model.gnb']||

### gpc_train


Component version: 1.0.0

Provide gaussian process classifier training. This component is currently experimental.
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|max_iter_predict|The maximum number of iterations in Newton's method for approximating the posterior during predict. Smaller values will reduce computation time at the cost of worse results.|Integer|N|Default: 20.Range: [1, $\infty$).|
|n_classes|The number of classes in the training data, must be preprocessed to 0, 1, 2, ...|Integer|N|Default: 2.Range: [1, $\infty$).|

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_ds|Input vertical table.|['sf.table.vertical', 'sf.table.individual']|Pleae fill in extra table attributes.|
|input/input_ds/feature_selects|which features should be used for training.|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. Min column number to select(inclusive): 1. |
|input/input_ds/label|Label of train dataset.|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. Min column number to select(inclusive): 1. Max column number to select(inclusive): 1. |

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_model|Output model.|['sf.model.gpc']||

### kmeans_train


Component version: 1.0.0

Provide kmeans training. This component is currently experimental.
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|n_clusters|Number of clusters.|Integer|Y|Range: [1, $\infty$).|
|max_iter|Number of iterations for kmeans training.|Integer|N|Default: 10.Range: [1, $\infty$).|
|n_init|Number of groups for initial centers.|Integer|N|Default: 1.Range: [1, $\infty$).|
|init_method|Params initialization method.|Special type. Union group. You must select one child to fill in.|N/A|This is a special type. This is a union group, you must select one child to fill in (if exists).|

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_ds|Input vertical table.|['sf.table.vertical']|Pleae fill in extra table attributes.|
|input/input_ds/feature_selects|which features should be used for training.|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. Min column number to select(inclusive): 1. |

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_model|Output model.|['sf.model.kmeans']||

### knn_train


Component version: 1.0.0

Provide k neighbors classifier training. This component is currently experimental.
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|weights|weights function used in prediction method.|Special type. Union group. You must select one child to fill in.|N/A|This is a special type. This is a union group, you must select one child to fill in (if exists).|
|n_classes|The number of classes in the training data, must be preprocessed to 0, 1, 2, ...|Integer|N|Default: 2.Range: [1, $\infty$).|
|n_neighbors|Number of neighbors to use for prediction.|Integer|N|Default: 5.Range: [1, $\infty$).|

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_ds|Input vertical table.|['sf.table.vertical']|Pleae fill in extra table attributes.|
|input/input_ds/feature_selects|which features should be used for training.|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. Min column number to select(inclusive): 1. |
|input/input_ds/label|Label of train dataset.|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. Min column number to select(inclusive): 1. Max column number to select(inclusive): 1. |

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_model|Output model.|['sf.model.knn']||

### sgb_train


Component version: 1.1.0

Provides both classification and regression tree boosting (also known as GBDT, GBM)
for vertical split dataset setting by using secure boost.
- SGB is short for SecureBoost. Compared to its safer counterpart SS-XGB, SecureBoost focused on protecting label holder.
- Check https://arxiv.org/abs/1901.08755.
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|num_boost_round|Number of boosting iterations.|Integer|N|Default: 10.Range: [1, $\infty$).|
|max_depth|Maximum depth of a tree.|Integer|N|Default: 5.Range: [1, 16].|
|learning_rate|Step size shrinkage used in update to prevent overfitting.|Float|N|Default: 0.1.Range: (0.0, 1.0].|
|objective|Specify the learning objective.|String|N|Default: logistic.Allowed: ['linear', 'logistic', 'tweedie'].|
|reg_lambda|L2 regularization term on weights.|Float|N|Default: 0.1.Range: [0.0, 10000.0].|
|gamma|Greater than 0 means pre-pruning enabled. If gain of a node is less than this value, it would be pruned.|Float|N|Default: 1.0.Range: [0.0, 10000.0].|
|colsample_by_tree|Subsample ratio of columns when constructing each tree.|Float|N|Default: 1.0.Range: (0.0, 1.0].|
|sketch_eps|This roughly translates into O(1 / sketch_eps) number of bins.|Float|N|Default: 0.1.Range: (0.0, 1.0].|
|base_score|The initial prediction score of all instances, global bias.|Float|N|Default: 0.0.Range: [-10.0, 10.0].|
|seed|Pseudorandom number generator seed.|Integer|N|Default: 42.Range: [0, $\infty$).|
|fixed_point_parameter|Any floating point number encoded by heu, will multiply a scale and take the round, scale = 2 ** fixed_point_parameter. larger value may mean more numerical accuracy, but too large will lead to overflow problem.|Integer|N|Default: 20.Range: [1, 100].|
|first_tree_with_label_holder_feature|Whether to train the first tree with label holder's own features.|Boolean|N|Default: False.|
|batch_encoding_enabled|If use batch encoding optimization.|Boolean|N|Default: True.|
|enable_quantization|Whether enable quantization of g and h.|Boolean|N|Default: False.|
|quantization_scale|Scale the sum of g to the specified value.|Float|N|Default: 10000.0.Range: [0.0, 10000000.0].|
|max_leaf|Maximum leaf of a tree. Only effective if train leaf wise.|Integer|N|Default: 15.Range: [1, 32768].|
|rowsample_by_tree|Row sub sample ratio of the training instances.|Float|N|Default: 1.0.Range: (0.0, 1.0].|
|enable_goss|Whether to enable GOSS.|Boolean|N|Default: False.|
|top_rate|GOSS-specific parameter. The fraction of large gradients to sample.|Float|N|Default: 0.3.Range: (0.0, 1.0].|
|bottom_rate|GOSS-specific parameter. The fraction of small gradients to sample.|Float|N|Default: 0.5.Range: (0.0, 1.0].|
|tree_growing_method|How to grow tree?|String|N|Default: level.|
|enable_early_stop|Whether to enable early stop during training.|Boolean|N|Default: False.|
|enable_monitor|Whether to enable monitoring performance during training.|Boolean|N|Default: False.|
|eval_metric|Use what metric for monitoring and early stop? Currently support ['roc_auc', 'rmse', 'mse', 'tweedie_deviance', 'tweedie_nll']|String|N|Default: roc_auc.Allowed: ['roc_auc', 'rmse', 'mse', 'tweedie_deviance', 'tweedie_nll'].|
|validation_fraction|Early stop specific parameter. Only effective if early stop enabled. The fraction of samples to use as validation set.|Float|N|Default: 0.1.Range: (0.0, 1.0).|
|stopping_rounds|Early stop specific parameter. If more than 'stopping_rounds' consecutive rounds without improvement, training will stop. Only effective if early stop enabled|Integer|N|Default: 1.Range: [1, 1024].|
|stopping_tolerance|Early stop specific parameter. If metric on validation set is no longer improving by at least this amount, then consider not improving.|Float|N|Default: 0.0.Range: [0.0, $\infty$).|
|tweedie_variance_power|Parameter that controls the variance of the Tweedie distribution.|Float|N|Default: 1.5.Range: (1.0, 2.0).|
|save_best_model|Whether to save the best model on validation set during training.|Boolean|N|Default: False.|
|report_importances|Whether to report feature importances. Currently supported importances are: {"gain": "the average gain across all splits the feature is used in.", "weight": "the number of times a feature is used to split the data across all trees."}|Boolean|N|Default: False.|

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_ds|Input vertical table.|['sf.table.vertical']|Pleae fill in extra table attributes.|
|input/input_ds/feature_selects|which features should be used for training.|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. Min column number to select(inclusive): 1. |
|input/input_ds/label|Label of train dataset.|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. Min column number to select(inclusive): 1. Max column number to select(inclusive): 1. |

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_model|Output model.|['sf.model.sgb']||
|report|If report_importances is true, report feature importances|['sf.report']||

### ss_glm_train


Component version: 1.1.0

generalized linear model (GLM) is a flexible generalization of ordinary linear regression.
The GLM generalizes linear regression by allowing the linear model to be related to the response
variable via a link function and by allowing the magnitude of the variance of each measurement to
be a function of its predicted value.
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|epochs|The number of complete pass through the training data.|Integer|N|Default: 10.Range: [1, $\infty$).|
|learning_rate|The step size at each iteration in one iteration.|Float|N|Default: 0.1.Range: (0.0, $\infty$).|
|batch_size|The number of training examples utilized in one iteration.|Integer|N|Default: 1024.Range: (0, $\infty$).|
|link_type|link function type|String|Y|Allowed: ['Logit', 'Log', 'Reciprocal', 'Identity'].|
|label_dist_type|label distribution type|String|Y|Allowed: ['Bernoulli', 'Poisson', 'Gamma', 'Tweedie'].|
|tweedie_power|Tweedie distribution power parameter|Float|N|Default: 1.0.Range: [0.0, 2.0].|
|dist_scale|A guess value for distribution's scale|Float|N|Default: 1.0.Range: [1.0, $\infty$).|
|iter_start_irls|run a few rounds of IRLS training as the initialization of w, 0 disable|Integer|N|Default: 0.Range: [0, $\infty$).|
|decay_epoch|decay learning interval|Integer|N|Default: 0.Range: [0, $\infty$).|
|decay_rate|decay learning rate|Float|N|Default: 0.0.Range: [0.0, 1.0).|
|optimizer|which optimizer to use: IRLS(Iteratively Reweighted Least Squares) or SGD(Stochastic Gradient Descent)|String|Y|Allowed: ['SGD', 'IRLS'].|
|l2_lambda|L2 regularization term|Float|N|Default: 0.1.Range: [0.0, $\infty$).|
|infeed_batch_size_limit|size of a single block, default to 8w * 100. increase the size will increase memory cost, but may decrease running time. Suggested to be as large as possible. (too large leads to OOM)|Integer|N|Default: 8000000.Range: [1000, 8000000].|
|fraction_of_validation_set|fraction of training set to be used as the validation set. ineffective for 'weight' stopping_metric|Float|N|Default: 0.2.Range: (0.0, 1.0).|
|random_state|random state for validation split|Integer|N|Default: 1212.Range: [0, $\infty$).|
|stopping_metric|use what metric as the condition for early stop? Must be one of ['deviance', 'MSE', 'RMSE', 'AUC', 'weight']. only logit link supports AUC metric (note that AUC is very, very expensive in MPC)|String|N|Default: deviance.Allowed: ['deviance', 'MSE', 'RMSE', 'AUC', 'weight'].|
|stopping_rounds|If the model is not improving for stopping_rounds, the training process will be stopped, for 'weight' stopping metric, stopping_rounds is fixed to be 1|Integer|N|Default: 0.Range: [0, 100].|
|stopping_tolerance|the model is considered as not improving, if the metric is not improved by tolerance over best metric in history. If metric is 'weight' and tolerance == 0, then early stop is disabled.|Float|N|Default: 0.001.Range: [0.0, 1.0).|
|report_metric|Whether to report the value of stopping metric. Only effective if early stop is enabled. If this option is set to true, metric will be revealed and logged.|Boolean|N|Default: False.|
|use_high_precision_exp|If you do not know the details of this parameter, please do not modify this parameter! If this option is true, glm training and prediction will use a high-precision exp approx, but there will be a large performance drop. Otherwise, use high performance exp approx, There will be no significant difference in model performance. However, prediction bias may occur if the model is exported to an external system for use.|Boolean|N|Default: False.|
|exp_mode|If you do not know the details of this parameter, please do not modify this parameter! Specify the mode of exp taylor approx, currently only supports 'taylor', 'pade' and 'prime' modes. The default value is 'taylor'. 'taylor': use taylor approx, variable precision and cost, higher exp_iters, higher cost. 'pade': use pade approx, high precision, high cost. 'prime': use prime approx, best precision, 3/4 cost of taylor (8 iter), only support for SEMI2K FM128 case. Although it has great presicion and performance inside valid domain, the approximation can be wildly inaccurate outside the valid domain. Suppose x -> exp(x), then valid domain is: x in ((47 - offset - 2fxp)/log_2(e), (125 - 2fxp - offset)/log_2(e)). That's why we need clamping x to this range. However, clamping action is expensive, so we need to set a reasonable offset to control the valid range of exp prime method, and avoid clamping for best performance.|String|N|Default: taylor.Allowed: ['pade', 'taylor', 'prime'].|
|exp_iters|If you do not know the details of this parameter, please do not modify this parameter! Specify the number of iterations of exp taylor approx, Only takes effect when using exp mode 'taylor'. Increasing this value will improve the accuracy of exp approx, but will quickly degrade performance.|Integer|N|Default: 8.Range: [4, 32].|
|exp_prime_offset|If you do not know the details of this parameter, please do not modify this parameter! Specify the offset of exp prime approx, only takes effect when using exp mode 'prime'. control the valid range of exp prime method. Suppose x -> exp(x), then valid domain is: x in ((47 - offset - 2fxp)/log_2(e), (125 - 2fxp - offset)/log_2(e)) default to be 13.|Integer|N|Default: 13.Range: (0, $\infty$).|
|exp_prime_lower_bound_clamp|If you do not know the details of this parameter, please do not modify this parameter! Specify whether to use lower bound for exp prime mode, only takes effect when using exp mode 'prime'. when calculating x -> exp(x), exp prime is only effective for x in ((47 - offset - 2fxp)/log_2(e), (125 - 2fxp - offset)/log_2(e)). If true, use clamp value below the lower bound, otherwise leave the value unchanged. lower bound is set to be (48 - offset - 2fxp)/log_2(e). Enable clamping will avoid large numerical errors when x < lower bound. Disable clamping will leave the value unchanged, which may cause large numerical errors when x < lower bound. However, clamping cost is very high, if we are certain x is in the valid range, it is recommended to disable clamping.|Boolean|N|Default: True.|
|exp_prime_higher_bound_clamp|If you do not know the details of this parameter, please do not modify this parameter! Specify whether to use upper bound for exp prime mode, only takes effect when using exp mode 'prime'. when calculating x -> exp(x), exp prime is only effective for x in ((47 - offset - 2fxp)/log_2(e), (125 - 2fxp - offset)/log_2(e)). If true, use clamp value above the upper bound, otherwise leave the value unchanged. upper bound is set to be (125 - 2fxp - offset)/log_2(e). Enable clamping will avoid large numerical errors when x > upper bound. Disable clamping will leave the value unchanged, which may cause large numerical errors when x > upper bound. However, clamping cost is very high, if we are certain x is in the valid range, it is recommended to disable clamping.|Boolean|N|Default: False.|
|report_weights|If this option is set to true, model will be revealed and model details are visible to all parties|Boolean|N|Default: False.|

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_ds|Input vertical table.|['sf.table.vertical']|Pleae fill in extra table attributes.|
|input/input_ds/feature_selects|which features should be used for training.|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. Min column number to select(inclusive): 1. |
|input/input_ds/offset|Specify a column to use as the offset|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. Max column number to select(inclusive): 1. |
|input/input_ds/weight|Specify a column to use for the observation weights|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. Max column number to select(inclusive): 1. |
|input/input_ds/label|Label of train dataset.|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. Min column number to select(inclusive): 1. Max column number to select(inclusive): 1. |

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_model|Output model.|['sf.model.ss_glm']||
|report|If report_weights is true, report model details|['sf.report']||

### ss_sgd_train


Component version: 1.0.0

Train both linear and logistic regression
linear models for vertical partitioning dataset with mini batch SGD training solver by using secret sharing.
- SS-SGD is short for secret sharing SGD training.
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|epochs|The number of complete pass through the training data.|Integer|N|Default: 10.Range: [1, $\infty$).|
|learning_rate|The step size at each iteration in one iteration.|Float|N|Default: 0.1.Range: (0.0, $\infty$).|
|batch_size|The number of training examples utilized in one iteration.|Integer|N|Default: 1024.Range: (0, $\infty$).|
|sig_type|Sigmoid approximation type.|String|N|Default: t1.Allowed: ['real', 't1', 't3', 't5', 'df', 'sr', 'mix'].|
|reg_type|Regression type|String|N|Default: logistic.Allowed: ['linear', 'logistic'].|
|penalty|The penalty(aka regularization term) to be used.|String|N|Default: None.Allowed: ['None', 'l2'].|
|l2_norm|L2 regularization term.|Float|N|Default: 0.5.Range: [0.0, $\infty$).|
|eps|If the change rate of weights is less than this threshold, the model is considered to be converged, and the training stops early. 0 to disable.|Float|N|Default: 0.001.Range: [0.0, $\infty$).|
|report_weights|If this option is set to true, model will be revealed and model details are visible to all parties|Boolean|N|Default: False.|

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_ds|Input vertical table.|['sf.table.vertical']|Pleae fill in extra table attributes.|
|input/input_ds/feature_selects|which features should be used for training.|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. Min column number to select(inclusive): 1. |
|input/input_ds/label|Label of train dataset.|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. Min column number to select(inclusive): 1. Max column number to select(inclusive): 1. |

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_model|Output model.|['sf.model.ss_sgd']||
|report|If report_weights is true, report model details|['sf.report']||

### ss_xgb_train


Component version: 1.0.0

This method provides both classification and regression tree boosting (also known as GBDT, GBM)
for vertical partitioning dataset setting by using secret sharing.
- SS-XGB is short for secret sharing XGB.
- More details: https://arxiv.org/pdf/2005.08479.pdf
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|num_boost_round|Number of boosting iterations.|Integer|N|Default: 10.Range: [1, $\infty$).|
|max_depth|Maximum depth of a tree.|Integer|N|Default: 5.Range: [1, 16].|
|learning_rate|Step size shrinkage used in updates to prevent overfitting.|Float|N|Default: 0.1.Range: (0.0, 1.0].|
|objective|Specify the learning objective.|String|N|Default: logistic.Allowed: ['linear', 'logistic'].|
|reg_lambda|L2 regularization term on weights.|Float|N|Default: 0.1.Range: [0.0, 10000.0].|
|subsample|Subsample ratio of the training instances.|Float|N|Default: 0.1.Range: (0.0, 1.0].|
|colsample_by_tree|Subsample ratio of columns when constructing each tree.|Float|N|Default: 0.1.Range: (0.0, 1.0].|
|sketch_eps|This roughly translates into O(1 / sketch_eps) number of bins.|Float|N|Default: 0.1.Range: (0.0, 1.0].|
|base_score|The initial prediction score of all instances, global bias.|Float|N|Default: 0.0.Range: [-10.0, 10.0].|
|seed|Pseudorandom number generator seed.|Integer|N|Default: 42.Range: [0, $\infty$).|

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_ds|Input vertical table.|['sf.table.vertical']|Pleae fill in extra table attributes.|
|input/input_ds/feature_selects|which features should be used for training.|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. Min column number to select(inclusive): 1. |
|input/input_ds/label|Label of train dataset.|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. Min column number to select(inclusive): 1. Max column number to select(inclusive): 1. |

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_model|Output model.|['sf.model.ss_xgb']||

## model

### model_export


Component version: 1.0.0

The model_export component supports converting and
packaging the rule files generated by preprocessing and
postprocessing components, as well as the model files generated
by model operators, into a Secretflow-Serving model package. The
list of components to be exported must contain exactly one model
train or model predict component, and may include zero or
multiple preprocessing and postprocessing components.
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|model_name|model's name|String|Y||
|model_desc|Describe what the model does|String|N|Default: .|
|input_datasets|The input data IDs for all components to be exported. Their order must remain consistent with the sequence in which the components were executed.|String List|Y||
|output_datasets|The output data IDs for all components to be exported. Their order must remain consistent with the sequence in which the components were executed.|String List|Y||
|component_eval_params|The eval parameters (in JSON format) for all components to be exported. Their order must remain consistent with the sequence in which the components were executed.|String List|Y||
|he_mode|If enabled, it will export a homomorphic encryption model. Currently, only SGD and GLM models for two-party scenarios are supported.|Boolean|N|Default: False.|

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_package|output tar package uri|['sf.serving.model']||
|report|report dumped model's input schemas|['sf.report']||

## postprocessing

### score_card_transformer


Component version: 1.0.0

Transform the predicted result (a probability value) produced by the logistic regression model into a more understandable score (for example, a score of up to 1000 points)
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|positive|Value for positive cases.|Integer|Y|Allowed: [0, 1].|
|predict_score_name||String|Y||
|scaled_value|Set a benchmark score that can be adjusted for specific business scenarios|Integer|Y|Range: (0, $\infty$).|
|odd_base|the odds value at given score baseline, odds = p / (1-p)|Float|Y|Range: (0.0, $\infty$).|
|pdo|points to double the odds|Float|Y|Range: (0.0, $\infty$).|
|min_score|An integer of [0,999] is supported|Integer|N|Default: 0.Range: [0, 999].|
|max_score|An integer of [1,1000] is supported|Integer|N|Default: 1000.Range: [1, 1000].|

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_ds|predict result table|['sf.table.individual']|Pleae fill in extra table attributes.|
|input/input_ds/predict_name||String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. Min column number to select(inclusive): 1. Max column number to select(inclusive): 1. |

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_ds|output table|['sf.table.individual']||

## preprocessing

### binary_op


Component version: 1.0.0

Perform binary operation binary_op(f1, f2) and assign the result to f3, f3 can be new or old. Currently f1, f2 and f3 all belong to a single party.
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|binary_op|What kind of binary operation we want to do, currently only supports +, -, *, /|String|N|Default: +.Allowed: ['+', '-', '*', '/'].|
|new_feature_name|Name of the newly generated feature. If this feature already exists, it will be overwritten.|String|Y||
|as_label|If True, the generated feature will be marked as label in schema, otherwise it will be treated as Feature.|Boolean|N|Default: False.|

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_ds|Input vertical table.|['sf.table.vertical']|Pleae fill in extra table attributes.|
|input/input_ds/f1|Feature 1 to operate on.|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. Min column number to select(inclusive): 1. Max column number to select(inclusive): 1. |
|input/input_ds/f2|Feature 2 to operate on.|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. Min column number to select(inclusive): 1. Max column number to select(inclusive): 1. |

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_ds|Output vertical table.|['sf.table.vertical']||
|output_rule|feature gen rule|['sf.rule.preprocessing']||

### case_when


Component version: 1.0.0

case_when
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|rules|input CaseWhen rules|Special type. SecretFlow customized Protocol Buffers message.|Y||

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_ds|Input vertical table.|['sf.table.vertical']||

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_ds|output_dataset|['sf.table.vertical']||
|output_rule|case when substitution rule|['sf.rule.preprocessing']||

### cast


Component version: 1.0.0

For conversion between basic data types, such as converting float to string.
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|astype|single-choice, options available are string, integer, float|Special type. Union group. You must select one child to fill in.|N/A|This is a special type. This is a union group, you must select one child to fill in (if exists).|

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_ds|The input table|['sf.table.vertical']|Pleae fill in extra table attributes.|
|input/input_ds/columns|Multiple-choice, options available are string, integer, float, boolean|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. Min column number to select(inclusive): 1. |

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_ds|The output table|['sf.table.vertical']||
|output_rule|The output rules|['sf.rule.preprocessing']||

### feature_calculate


Component version: 1.0.0

Generate a new feature by performing calculations on an origin feature
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|rules|input CalculateOpRules rules|Special type. SecretFlow customized Protocol Buffers message.|Y||

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_ds|Input vertical table|['sf.table.vertical']|Pleae fill in extra table attributes.|
|input/input_ds/features|Feature(s) to operate on|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. Min column number to select(inclusive): 1. |

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_ds|output_dataset|['sf.table.vertical']||
|output_rule|feature calculate rule|['sf.rule.preprocessing']||

### fillna


Component version: 1.0.0

Fill null/nan or other specificed outliers in dataset
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|nan_is_null|Whether floating-point NaN values are considered null, take effect with float columns|Boolean|N|Default: True.|
|float_outliers|These outlier value are considered null, take effect with float columns|Float List|N|Default: [].|
|int_outliers|These outlier value are considered null, take effect with int columns|Integer List|N|Default: [].|
|str_outliers|These outlier value are considered null, take effect with str columns|String List|N|Default: [].|
|str_fill_strategy|Replacement strategy for str column. If "most_frequent", then replace missing using the most frequent value along each column. If "constant", then replace missing values with fill_value_str.|String|N|Default: constant.Allowed: ['constant', 'most_frequent'].|
|fill_value_str|For str type data. If method is 'constant' use this value for filling null.|String|N|Default: .|
|int_fill_strategy|Replacement strategy for int column. If "mean", then replace missing values using the mean along each column. If "median", then replace missing values using the median along each column If "most_frequent", then replace missing using the most frequent value along each column. If "constant", then replace missing values with fill_value_int.|String|N|Default: constant.Allowed: ['mean', 'median', 'most_frequent', 'constant'].|
|fill_value_int|For int type data. If method is 'constant' use this value for filling null.|Integer|N|Default: 0.|
|float_fill_strategy|Replacement strategy for float column. If "mean", then replace missing values using the mean along each column. If "median", then replace missing values using the median along each column If "most_frequent", then replace missing using the most frequent value along each column. If "constant", then replace missing values with fill_value_float.|String|N|Default: constant.Allowed: ['mean', 'median', 'most_frequent', 'constant'].|
|fill_value_float|For float type data. If method is 'constant' use this value for filling null.|Float|N|Default: 0.0.|
|bool_fill_strategy|Replacement strategy for bool column. If "most_frequent", then replace missing using the most frequent value along each column. If "constant", then replace missing values with fill_value_bool.|String|N|Default: constant.Allowed: ['constant', 'most_frequent'].|
|fill_value_bool|For bool type data. If method is 'constant' use this value for filling null.|Boolean|N|Default: False.|

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_ds|Input vertical table.|['sf.table.vertical']|Pleae fill in extra table attributes.|
|input/input_ds/fill_na_features|Features to fill.|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. |

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_ds|Output vertical table.|['sf.table.vertical']||
|output_rule|fill value rule|['sf.rule.preprocessing']||

### onehot_encode


Component version: 1.0.0

onehot_encode
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|drop|drop unwanted category based on selection|Special type. Union group. You must select one child to fill in.|N/A|This is a special type. This is a union group, you must select one child to fill in (if exists).|
|min_frequency|Specifies the minimum frequency below which a category will be considered infrequent, [0, 1), 0 disable|Float|N|Default: 0.0.Range: [0.0, 1.0).|
|report_rules|Whether to report rule details|Boolean|N|Default: True.|

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_ds|Input vertical table.|['sf.table.vertical']|Pleae fill in extra table attributes.|
|input/input_ds/features|Features to encode.|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. |

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_ds|output dataset|['sf.table.vertical']||
|output_rule|onehot rule|['sf.rule.preprocessing']||
|report|report rules details if report_rules is true|['sf.report']||

### sql_processor


Component version: 1.0.0

sql processor
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|sql|sql for preprocessing, for example SELECT a, b, a+b|String|Y||

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_ds|Input table|['sf.table.individual', 'sf.table.vertical']||

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_ds|Output table|['sf.table.individual', 'sf.table.vertical']||
|output_rule|Output rule|['sf.rule.preprocessing']||

### substitution


Component version: 1.0.0

unified substitution component
#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_ds|Input vertical table.|['sf.table.vertical']||
|input_rule|Input preprocessing rules|['sf.rule.preprocessing', 'sf.rule.binning']||

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_ds|output_dataset|['sf.table.vertical']||

### vert_binning


Component version: 1.0.0

Generate equal frequency or equal range binning rules for vertical partitioning datasets.
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|binning_method|How to bin features with numeric types: "quantile"(equal frequency)/"eq_range"(equal range)|String|N|Default: eq_range.Allowed: ['eq_range', 'quantile'].|
|bin_num|Max bin counts for one features.|Integer|N|Default: 10.Range: [2, $\infty$).|
|report_rules|Whether report binning rules.|Boolean|N|Default: False.|

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_ds|Input vertical table.|['sf.table.vertical']|Pleae fill in extra table attributes.|
|input/input_ds/feature_selects|which features should be binned.|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. Min column number to select(inclusive): 1. |

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_ds|Output vertical table.|['sf.table.vertical']||
|output_rule|Output bin rule.|['sf.rule.binning']||
|report|report rules details if report_rules is true|['sf.report']||

### vert_woe_binning


Component version: 1.0.0

Generate Weight of Evidence (WOE) binning rules for vertical partitioning datasets.
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|secure_device_type|Use SPU(Secure multi-party computation or MPC) or HEU(Homomorphic encryption or HE) to secure bucket summation.|String|N|Default: spu.Allowed: ['spu', 'heu'].|
|binning_method|How to bin features with numeric types: "quantile"(equal frequency)/"chimerge"(ChiMerge from AAAI92-019: https://www.aaai.org/Papers/AAAI/1992/AAAI92-019.pdf)/"eq_range"(equal range)|String|N|Default: quantile.Allowed: ['quantile', 'chimerge', 'eq_range'].|
|bin_num|Max bin counts for one features.|Integer|N|Default: 10.Range: (0, $\infty$).|
|positive_label|Which value represent positive value in label.|String|N|Default: 1.|
|chimerge_init_bins|Max bin counts for initialization binning in ChiMerge.|Integer|N|Default: 100.Range: (2, $\infty$).|
|chimerge_target_bins|Stop merging if remaining bin counts is less than or equal to this value.|Integer|N|Default: 10.Range: [2, $\infty$).|
|chimerge_target_pvalue|Stop merging if biggest pvalue of remaining bins is greater than this value.|Float|N|Default: 0.1.Range: (0.0, 1.0].|
|report_rules|Whether report binning rules.|Boolean|N|Default: False.|

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_ds|Input vertical table.|['sf.table.vertical']|Pleae fill in extra table attributes.|
|input/input_ds/feature_selects|which features should be binned. WARNING: WOE won't be effective for features with enumeration count <=2.|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. Min column number to select(inclusive): 1. |
|input/input_ds/label|Label of input data.|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. Min column number to select(inclusive): 1. Max column number to select(inclusive): 1. |

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_ds|Output vertical table.|['sf.table.vertical']||
|output_rule|Output WOE rule.|['sf.rule.binning']||
|report|report rules details if report_rules is true|['sf.report']||

## stats

### groupby_statistics


Component version: 1.0.0

Get a groupby of statistics, like pandas groupby statistics.
Currently only support VDataframe.
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|aggregation_config|input groupby aggregation config|Special type. SecretFlow customized Protocol Buffers message.|Y||
|max_group_size|The maximum number of groups allowed|Integer|N|Default: 10000.Range: (0, 10001).|

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_ds|Input table.|['sf.table.vertical', 'sf.table.individual']|Pleae fill in extra table attributes.|
|input/input_ds/by|by what columns should we group the values, encode values into int or str before groupby or else numeric errors may occur|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. Min column number to select(inclusive): 1. Max column number to select(inclusive): 4. |

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|report|Output groupby statistics report.|['sf.report']||

### ss_pearsonr


Component version: 1.0.0

Calculate Pearson's product-moment correlation coefficient for vertical partitioning dataset
by using secret sharing.
- For large dataset(large than 10w samples & 200 features), recommend to use [Ring size: 128, Fxp: 40] options for SPU device.
#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_ds|Input vertical table.|['sf.table.vertical', 'sf.table.individual']|Pleae fill in extra table attributes.|
|input/input_ds/feature_selects|Specify which features to calculate correlation coefficient with. If empty, all features will be used|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. |

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|report|Output Pearson's product-moment correlation coefficient report.|['sf.report']||

### ss_vif


Component version: 1.0.0

Calculate Variance Inflation Factor(VIF) for vertical partitioning dataset
by using secret sharing.
- For large dataset(large than 10w samples & 200 features), recommend to use [Ring size: 128, Fxp: 40] options for SPU device.
#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_ds|Input vertical table.|['sf.table.vertical', 'sf.table.individual']|Pleae fill in extra table attributes.|
|input/input_ds/feature_selects|Specify which features to calculate VIF with. If empty, all features will be used.|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. |

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|report|Output Variance Inflation Factor(VIF) report.|['sf.report']||

### stats_psi


Component version: 1.0.0

population stability index.
#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_base_ds|Input base vertical table.|['sf.table.vertical', 'sf.table.individual']|Pleae fill in extra table attributes.|
|input/input_base_ds/feature_selects|which features should be binned.|String List(Set value with other Component Attributes)|You need to select some columns of table input_base_ds. Min column number to select(inclusive): 1. |
|input_test_ds|Input test vertical table.|['sf.table.vertical', 'sf.table.individual']||
|input_rule|Input bin rule.|['sf.rule.binning']||

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|report|Output population stability index.|['sf.report']||

### table_statistics


Component version: 1.0.2

Get a table of statistics,
including each column's
1. datatype
2. total_count
3. count
4. count_na
5. na_ratio
6. min
7. max
8. mean
9. var
10. std
11. sem
12. skewness
13. kurtosis
14. q1
15. q2
16. q3
17. moment_2
18. moment_3
19. moment_4
20. central_moment_2
21. central_moment_3
22. central_moment_4
23. sum
24. sum_2
25. sum_3
26. sum_4
- moment_2 means E[X^2].
- central_moment_2 means E[(X - mean(X))^2].
- sum_2 means sum(X^2).
All of the object or string class columns will not be included in the above statistics, but in a separate report.
The second report is a table of the object or string class columns.
Note that please do not include individual information (like address, phone number, etc.) for table statistics.
The categorical report will be with the following columns:
1. column dtype (the data type of the column)
2. count (the number of non-null values)
3. nunique (the number of unique values in this column)
if no numeric or categorical columns, the report will be dummy report.
#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_ds|Input table.|['sf.table.vertical', 'sf.table.individual']|Pleae fill in extra table attributes.|
|input/input_ds/features|perform statistics on these columns|String List(Set value with other Component Attributes)|You need to select some columns of table input_ds. Min column number to select(inclusive): 1. |

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|report|Output table statistics report.|['sf.report']||
