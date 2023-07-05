



SecretFlow Component List
=========================


Last update: Mon Jul  3 08:44:11 2023

Version: 0.0.1

First-party SecretFlow components.
## feature

### vert_woe_binning


Component version: 0.0.1

Generate Weight of Evidence (WOE) binning rules for vertical partitioning datasets.
#### Attrs


|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|secure_device_type|Use SPU or HEU to secure bucket summation.|String|N|Default: spu. Allowed: ['spu', 'heu'].|
|binning_method|How to bin features with numeric types: "quantile"(equal frequency)/"chimerge"(ChiMerge from AAAI92-019: https://www.aaai.org/Papers/AAAI/1992/AAAI92-019.pdf)|String|N|Default: quantile. Allowed: ['quantile', 'chimerge'].|
|bin_num|Max bin counts for one features.|Integer|N|Default: 10. Range: (0, $\infty$).|
|positive_label|Which value represent positive value in label.|String|N|Default: 1.|
|chimerge_init_bins|Max bin counts for initialization binning in ChiMerge.|Integer|N|Default: 100. Range: (2, $\infty$).|
|chimerge_target_bins|Stop merging if remaining bin counts is less than or equal to this value.|Integer|N|Default: 10. Range: [2, $\infty$).|
|chimerge_target_pvalue|Stop merging if biggest pvalue of remaining bins is greater than this value.|Float|N|Default: 0.1. Range: (0.0, 1.0].|

#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_data|Input dataset for generating rule.|['sf.table.vertical_table']|Extra table attributes.(0) feature_selects - which features should be binned. Min column number to select(inclusive): 1. |

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|woe_rule|Output WOE rule.|['sf.rule.woe_binning']||

### vert_woe_substitution


Component version: 0.0.1

Substitute datasets' value by WOE substitution rules.
#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_data|Vertical partitioning dataset to be substituted.|['sf.table.vertical_table']||
|woe_rule|WOE substitution rule.|['sf.rule.woe_binning']||

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_data|Output substituted dataset.|['sf.table.vertical_table']||

## ml.eval

### biclassification_eval


Component version: 0.0.1

Statistics evaluation for a bi-classification model on a dataset. 1. summary_report: SummaryReport 2. group_reports: List[GroupReport] 3. eq_frequent_bin_report: List[EqBinReport] 4. eq_range_bin_report: List[EqBinReport] 5. head_report: List[PrReport] reports for fpr = 0.001, 0.005, 0.01, 0.05, 0.1, 0.2
#### Attrs


|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|bucket_size|Number of buckets.|Integer|N|Default: 10. Range: [1, $\infty$).|
|min_item_cnt_per_bucket|Min item cnt per bucket. If any bucket doesn't meet the requirement, error raises. For security reasons, we require this parameter to be at least 5.|Integer|N|Default: 5. Range: [5, $\infty$).|

#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|labels|labels|['sf.table.vertical_table', 'sf.table.individual']|Extra table attributes.(0) col - The column name to use in the dataset. If not provided, the label of dataset will be used by default. Max column number to select(inclusive): 1. |
|predictions|predictions|['sf.table.vertical_table', 'sf.table.individual']|Extra table attributes.(0) col - The column name to use in the dataset. If not provided, the label of dataset will be used by default. Max column number to select(inclusive): 1. |

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|reports|Output report.|['sf.report']||

### prediction_bias_eval


Component version: 0.0.1

Calculate prediction bias, ie. average of predictions - average of labels.
#### Attrs


|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|bucket_num|Num of bucket.|Integer|N|Default: 10. Range: [1, $\infty$).|
|min_item_cnt_per_bucket|Min item cnt per bucket. If any bucket doesn't meet the requirement, error raises. For security reasons, we require this parameter to be at least 2.|Integer|N|Default: 2. Range: [2, $\infty$).|
|bucket_method|Bucket method.|String|N|Default: equal_width. Allowed: ['equal_width', 'equal_frequency'].|

#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|labels|labels|['sf.table.vertical_table', 'sf.table.individual']|Extra table attributes.(0) col - The column name to use in the dataset. If not provided, the label of dataset will be used by default. Max column number to select(inclusive): 1. |
|predictions|predictions|['sf.table.vertical_table', 'sf.table.individual']|Extra table attributes.(0) col - The column name to use in the dataset. If not provided, the label of dataset will be used by default. Max column number to select(inclusive): 1. |

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|result|Output report.|['sf.report']||

### ss_pvalue


Component version: 0.0.1

Calculate P-Value for LR model training on vertical partitioning dataset by using secret sharing. For large dataset(large than 10w samples & 200 features), recommend to use [Ring size: 128, Fxp: 40] options for SPU device.
#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|model|Input model.|['sf.model.ss_sgd']||
|input_data|Input dataset.|['sf.table.vertical_table']||

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|report|Output report.|['sf.report']||

## ml.predict

### sgb_predict


Component version: 0.0.1

Predict using SGB model.
#### Attrs


|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|receiver|Party of receiver.|String|Y|Default: .|
|pred_name|Name for prediction column|String|N|Default: pred.|
|save_ids|Whether to save ids columns into output prediction table. If true, input feature_dataset must contain id columns, and receiver party must be id owner.|Boolean|N|Default: False.|
|save_label|Whether or not to save real label columns into output pred file. If true, input feature_dataset must contain label columns and receiver party must be label owner.|Boolean|N|Default: False.|

#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|model|model|['sf.model.sgb']||
|feature_dataset|Input feature dataset.|['sf.table.vertical_table']||

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|pred|Output prediction|['sf.table.individual']||

### ss_sgd_predict


Component version: 0.0.1

Predict using the SS-SGD model.
#### Attrs


|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|batch_size|The number of training examples utilized in one iteration.|Integer|N|Default: 1024. Range: (0, $\infty$).|
|receiver|Party of receiver.|String|Y|Default: .|
|pred_name|Column name for predictions.|String|N|Default: pred.|
|save_ids|Whether to save ids columns into output prediction table. If true, input feature_dataset must contain id columns, and receiver party must be id owner.|Boolean|N|Default: False.|
|save_label|Whether or not to save real label columns into output pred file. If true, input feature_dataset must contain label columns and receiver party must be label owner.|Boolean|N|Default: False.|

#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|model|Input model.|['sf.model.ss_sgd']||
|feature_dataset|Input feature dataset.|['sf.table.vertical_table']||

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|pred|Output prediction.|['sf.table.individual']||

### ss_xgb_predict


Component version: 0.0.1

Predict using the SS-XGB model.
#### Attrs


|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|receiver|Party of receiver.|String|Y|Default: .|
|pred_name|Column name for predictions.|String|N|Default: pred.|
|save_ids|Whether to save ids columns into output prediction table. If true, input feature_dataset must contain id columns, and receiver party must be id owner.|Boolean|N|Default: False.|
|save_label|Whether or not to save real label columns into output pred file. If true, input feature_dataset must contain label columns and receiver party must be label owner.|Boolean|N|Default: False.|

#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|model|Input model.|['sf.model.ss_xgb']||
|feature_dataset|Input features.|['sf.table.vertical_table']||

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|pred|Output prediction.|['sf.table.individual']||

## ml.train

### sgb_train


Component version: 0.0.1

Provides both classification and regression tree boosting (also known as GBDT, GBM) for vertical split dataset setting by using secure boost. SGB is short for SecureBoost. Compared to its safer counterpart SS-XGB, SecureBoost focused on protecting label holder. Check https://arxiv.org/abs/1901.08755.
#### Attrs


|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|num_boost_round|Number of boosting iterations.|Integer|N|Default: 10. Range: [1, $\infty$).|
|max_depth|Maximum depth of a tree.|Integer|N|Default: 5. Range: [1, 16].|
|learning_rate|Step size shrinkage used in update to prevent overfitting.|Float|N|Default: 0.1. Range: (0.0, 1.0].|
|objective|Specify the learning objective.|String|N|Default: logistic. Allowed: ['linear', 'logistic'].|
|reg_lambda|L2 regularization term on weights.|Float|N|Default: 0.1. Range: [0.0, 10000.0].|
|gamma|Greater than 0 means pre-pruning enabled. If gain of a node is less than this value, it would be pruned.|Float|N|Default: 0.1. Range: [0.0, 10000.0].|
|subsample|Subsample ratio of the training instances.|Float|N|Default: 1.0. Range: (0.0, 1.0].|
|colsample_by_tree|Subsample ratio of columns when constructing each tree.|Float|N|Default: 1.0. Range: (0.0, 1.0].|
|sketch_eps|This roughly translates into O(1 / sketch_eps) number of bins.|Float|N|Default: 0.1. Range: (0.0, 1.0].|
|base_score|The initial prediction score of all instances, global bias.|Float|N|Default: 0.0. Range: [0.0, $\infty$).|
|seed|Pseudorandom number generator seed.|Integer|N|Default: 42. Range: [0, $\infty$).|
|fixed_point_parameter|Any floating point number encoded by heu, will multiply a scale and take the round, scale = 2 ** fixed_point_parameter. larger value may mean more numerical accuracy, but too large will lead to overflow problem.|Integer|N|Default: 20. Range: [0, $\infty$).|

#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|train_dataset|Input train dataset.|['sf.table.vertical_table']||

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_model|Output model.|['sf.model.sgb']||

### ss_sgd_train


Component version: 0.0.1

Train both linear and logistic regression linear models for vertical partitioning dataset with mini batch SGD training solver by using secret sharing. SS-SGD is short for secret sharing SGD training.
#### Attrs


|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|epochs|The number of complete pass through the training data.|Integer|N|Default: 10. Range: [1, $\infty$).|
|learning_rate|The step size at each iteration in one iteration.|Float|N|Default: 0.1. Range: (0.0, $\infty$).|
|batch_size|The number of training examples utilized in one iteration.|Integer|N|Default: 1024. Range: (0, $\infty$).|
|sig_type|Sigmoid approximation type.|String|N|Default: t1. Allowed: ['real', 't1', 't3', 't5', 'df', 'sr', 'mix'].|
|reg_type|Regression type|String|N|Default: logistic. Allowed: ['linear', 'logistic'].|
|penalty|The penalty(aka regularization term) to be used.|String|N|Default: None. Allowed: ['None', 'l1', 'l2'].|
|l2_norm|L2 regularization term.|Float|N|Default: 0.5. Range: [0.0, $\infty$).|
|eps|If the change rate of weights is less than this threshold, the model is considered to be converged, and the training stops early. 0 to disable.|Float|N|Default: 0.001. Range: (0.0, $\infty$).|

#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|train_dataset|Input train dataset.|['sf.table.vertical_table']||

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_model|Output model|['sf.model.ss_sgd']||

### ss_xgb_train


Component version: 0.0.1

This method provides both classification and regression tree boosting (also known as GBDT, GBM) for vertical partitioning dataset setting by using secret sharing. SS-XGB is short for secret sharing XGB. More details: https://arxiv.org/pdf/2005.08479.pdf
#### Attrs


|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|num_boost_round|Number of boosting iterations.|Integer|N|Default: 10. Range: [1, $\infty$).|
|max_depth|Maximum depth of a tree.|Integer|N|Default: 5. Range: [1, 16].|
|learning_rate|Step size shrinkage used in updates to prevent overfitting.|Float|N|Default: 0.1. Range: (0.0, 1.0].|
|objective|Specify the learning objective.|String|N|Default: logistic. Allowed: ['linear', 'logistic'].|
|reg_lambda|L2 regularization term on weights.|Float|N|Default: 0.1. Range: [0.0, 10000.0].|
|subsample|Subsample ratio of the training instances.|Float|N|Default: 0.1. Range: (0.0, 1.0].|
|colsample_by_tree|Subsample ratio of columns when constructing each tree.|Float|N|Default: 0.1. Range: (0.0, 1.0].|
|sketch_eps|This roughly translates into O(1 / sketch_eps) number of bins.|Float|N|Default: 0.1. Range: (0.0, 1.0].|
|base_score|The initial prediction score of all instances, global bias.|Float|N|Default: 0.0. Range: [0.0, $\infty$).|
|seed|Pseudorandom number generator seed.|Integer|N|Default: 42. Range: [0, $\infty$).|

#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|train_dataset|Train dataset|['sf.table.vertical_table']||

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_model|Output model.|['sf.model.ss_xgb']||

## preprocessing

### feature_filter


Component version: 0.0.1

Drop features from the dataset.
#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|in_ds|Input dataset.|['sf.table.vertical_table']|Extra table attributes.(0) drop_features - Features to drop. |

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|out_ds|Output dataset with filtered features.|['sf.table.vertical_table']||

### psi


Component version: 0.0.1

Balanced PSI between two parties.
#### Attrs


|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|protocol|PSI protocol.|String|N|Default: ECDH_PSI_2PC. Allowed: ['ECDH_PSI_2PC', 'KKRT_PSI_2PC', 'BC22_PSI_2PC'].|
|bucket_size|Specify the hash bucket size used in PSI. Larger values consume more memory.|Integer|N|Default: 1048576. Range: (0, $\infty$).|
|ecdh_curve_type|Curve type for ECDH PSI.|String|N|Default: CURVE_FOURQ. Allowed: ['CURVE_25519', 'CURVE_FOURQ', 'CURVE_SM2', 'CURVE_SECP256K1'].|

#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|receiver_input|Input for receiver|['sf.table.individual']|Extra table attributes.(0) key - Column(s) used to join. If not provided, ids of the dataset will be used. |
|sender_input|Input for sender|['sf.table.individual']|Extra table attributes.(0) key - Column(s) used to join. If not provided, ids of the dataset will be used. |

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|psi_output|Output|['sf.table.vertical_table']||

### train_test_split


Component version: 0.0.1

Split datasets into random train and test subsets. Please check: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
#### Attrs


|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|train_size|Proportion of the dataset to include in the train subset.|Float|N|Default: 0.75. Range: [0.0, 1.0].|
|test_size|Proportion of the dataset to include in the test subset.|Float|N|Default: 0.25. Range: [0.0, 1.0].|
|random_state|Specify the random seed of the shuffling.|Integer|N|Default: 1024. Range: (0, $\infty$).|
|shuffle|Whether to shuffle the data before splitting.|Boolean|N|Default: True.|

#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_data|Input dataset.|['sf.table.vertical_table']||

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|train|Output train dataset.|['sf.table.vertical_table']||
|test|Output test dataset.|['sf.table.vertical_table']||

## stats

### ss_pearsonr


Component version: 0.0.1

Calculate Pearson's product-moment correlation coefficient for vertical partitioning dataset by using secret sharing. For large dataset(large than 10w samples & 200 features), recommend to use [Ring size: 128, Fxp: 40] options for SPU device.
#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_data|Input dataset.|['sf.table.vertical_table']|Extra table attributes.(0) feature_selects - Specify which features to calculate correlation coefficient with. If empty, all features will be used |

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|report|Output report.|['sf.report']||

### ss_vif


Component version: 0.0.1

Calculate Variance Inflation Factor(VIF) for vertical partitioning dataset by using secret sharing. For large dataset(large than 10w samples & 200 features), recommend to use [Ring size: 128, Fxp: 40] options for SPU device.
#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_data|Input dataset.|['sf.table.vertical_table']|Extra table attributes.(0) feature_selects - Specify which features to calculate VIF with. If empty, all features will be used. |

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|report|Output report.|['sf.report']||

### table_statistics


Component version: 0.0.1

Get a table of statistics, including each column's datatype, total_count, count, count_na, min, max, var, std, sem, skewness, kurtosis, q1, q2, q3, moment_2, moment_3, moment_4, central_moment_2, central_moment_3, central_moment_4, sum, sum_2, sum_3 and sum_4. moment_2 means E[X^2]. central_moment_2 means E[(X - mean(X))^2]. sum_2 means sum(X^2).
#### Inputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_data|Input data.|['sf.table.vertical_table', 'sf.table.individual']||

#### Outputs


|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|report|Output report.|['sf.report']||
