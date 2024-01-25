



SecretFlow Component List
=========================


Last update: Sat Oct 14 16:41:07 2023

Version: 0.0.1

First-party SecretFlow components.
## feature

### vert_bin_substitution


Component version: 0.0.1

Substitute datasets' value by bin substitution rules.
#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_data|Vertical partitioning dataset to be substituted.|['sf.table.vertical_table']||
|bin_rule|Input bin substitution rule.|['sf.rule.binning']||

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_data|Output vertical table.|['sf.table.vertical_table']||

### vert_binning


Component version: 0.0.1

Generate equal frequency or equal range binning rules for vertical partitioning datasets.
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|binning_method|How to bin features with numeric types: "quantile"(equal frequency)/"eq_range"(equal range)|String|N|Default: eq_range. Allowed: ['eq_range', 'quantile'].|
|bin_num|Max bin counts for one features.|Integer|N|Default: 10. Range: (0, $\infty$).|

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_data|Input vertical table.|['sf.table.vertical_table']|Extra table attributes.(0) feature_selects - which features should be binned. Min column number to select(inclusive): 1. |

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|bin_rule|Output bin rule.|['sf.rule.binning']||

### vert_woe_binning


Component version: 0.0.1

Generate Weight of Evidence (WOE) binning rules for vertical partitioning datasets.
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|secure_device_type|Use SPU(Secure multi-party computation or MPC) or HEU(Homomorphic encryption or HE) to secure bucket summation.|String|N|Default: spu. Allowed: ['spu', 'heu'].|
|binning_method|How to bin features with numeric types: "quantile"(equal frequency)/"chimerge"(ChiMerge from AAAI92-019: https://www.aaai.org/Papers/AAAI/1992/AAAI92-019.pdf)|String|N|Default: quantile. Allowed: ['quantile', 'chimerge'].|
|bin_num|Max bin counts for one features.|Integer|N|Default: 10. Range: (0, $\infty$).|
|positive_label|Which value represent positive value in label.|String|N|Default: 1.|
|chimerge_init_bins|Max bin counts for initialization binning in ChiMerge.|Integer|N|Default: 100. Range: (2, $\infty$).|
|chimerge_target_bins|Stop merging if remaining bin counts is less than or equal to this value.|Integer|N|Default: 10. Range: [2, $\infty$).|
|chimerge_target_pvalue|Stop merging if biggest pvalue of remaining bins is greater than this value.|Float|N|Default: 0.1. Range: (0.0, 1.0].|

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_data|Input vertical table.|['sf.table.vertical_table']|Extra table attributes.(0) feature_selects - which features should be binned. Min column number to select(inclusive): 1. |

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|bin_rule|Output WOE rule.|['sf.rule.binning']||

## ml.eval

### biclassification_eval


Component version: 0.0.1

Statistics evaluation for a bi-classification model on a dataset.
1. summary_report: SummaryReport
2. group_reports: List[GroupReport]
3. eq_frequent_bin_report: List[EqBinReport]
4. eq_range_bin_report: List[EqBinReport]
5. head_report: List[PrReport]
reports for fpr = 0.001, 0.005, 0.01, 0.05, 0.1, 0.2
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|bucket_size|Number of buckets.|Integer|N|Default: 10. Range: [1, $\infty$).|
|min_item_cnt_per_bucket|Min item cnt per bucket. If any bucket doesn't meet the requirement, error raises. For security reasons, we require this parameter to be at least 5.|Integer|N|Default: 5. Range: [5, $\infty$).|

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|labels|Input table with labels|['sf.table.vertical_table', 'sf.table.individual']|Extra table attributes.(0) col - The column name to use in the dataset. If not provided, the label of dataset will be used by default. Max column number to select(inclusive): 1. |
|predictions|Input table with predictions|['sf.table.vertical_table', 'sf.table.individual']|Extra table attributes.(0) col - The column name to use in the dataset. If not provided, the label of dataset will be used by default. Max column number to select(inclusive): 1. |

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
|labels|Input table with labels.|['sf.table.vertical_table', 'sf.table.individual']|Extra table attributes.(0) col - The column name to use in the dataset. If not provided, the label of dataset will be used by default. Max column number to select(inclusive): 1. |
|predictions|Input table with predictions.|['sf.table.vertical_table', 'sf.table.individual']|Extra table attributes.(0) col - The column name to use in the dataset. If not provided, the label of dataset will be used by default. Max column number to select(inclusive): 1. |

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|result|Output report.|['sf.report']||

### ss_pvalue


Component version: 0.0.1

Calculate P-Value for LR model training on vertical partitioning dataset by using secret sharing.
For large dataset(large than 10w samples & 200 features),
recommend to use [Ring size: 128, Fxp: 40] options for SPU device.
#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|model|Input model.|['sf.model.ss_sgd']||
|input_data|Input vertical table.|['sf.table.vertical_table']||

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|report|Output P-Value report.|['sf.report']||

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
|feature_dataset|Input vertical table.|['sf.table.vertical_table']||

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|pred|Output prediction.|['sf.table.individual']||

### ss_glm_predict


Component version: 0.0.1

Predict using the SSGLM model.
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|receiver|Party of receiver.|String|Y|Default: .|
|pred_name|Column name for predictions.|String|N|Default: pred.|
|save_ids|Whether to save ids columns into output prediction table. If true, input feature_dataset must contain id columns, and receiver party must be id owner.|Boolean|N|Default: False.|
|save_label|Whether or not to save real label columns into output pred file. If true, input feature_dataset must contain label columns and receiver party must be label owner.|Boolean|N|Default: False.|
|offset_col|Specify a column to use as the offset|String|N|Default: .|

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|model|Input model.|['sf.model.ss_glm']||
|feature_dataset|Input vertical table.|['sf.table.vertical_table']||

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|pred|Output prediction.|['sf.table.individual']||

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
|feature_dataset|Input vertical table.|['sf.table.vertical_table']||

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
|feature_dataset|Input vertical table.|['sf.table.vertical_table']||

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|pred|Output prediction.|['sf.table.individual']||

## ml.train

### sgb_train


Component version: 0.0.1

Provides both classification and regression tree boosting (also known as GBDT, GBM)
for vertical split dataset setting by using secure boost.
- SGB is short for SecureBoost. Compared to its safer counterpart SS-XGB, SecureBoost focused on protecting label holder.
- Check https://arxiv.org/abs/1901.08755.
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|num_boost_round|Number of boosting iterations.|Integer|N|Default: 10. Range: [1, $\infty$).|
|max_depth|Maximum depth of a tree.|Integer|N|Default: 5. Range: [1, 16].|
|learning_rate|Step size shrinkage used in update to prevent overfitting.|Float|N|Default: 0.1. Range: (0.0, 1.0].|
|objective|Specify the learning objective.|String|N|Default: logistic. Allowed: ['linear', 'logistic'].|
|reg_lambda|L2 regularization term on weights.|Float|N|Default: 0.1. Range: [0.0, 10000.0].|
|gamma|Greater than 0 means pre-pruning enabled. If gain of a node is less than this value, it would be pruned.|Float|N|Default: 0.1. Range: [0.0, 10000.0].|
|colsample_by_tree|Subsample ratio of columns when constructing each tree.|Float|N|Default: 1.0. Range: (0.0, 1.0].|
|sketch_eps|This roughly translates into O(1 / sketch_eps) number of bins.|Float|N|Default: 0.1. Range: (0.0, 1.0].|
|base_score|The initial prediction score of all instances, global bias.|Float|N|Default: 0.0. Range: [0.0, $\infty$).|
|seed|Pseudorandom number generator seed.|Integer|N|Default: 42. Range: [0, $\infty$).|
|fixed_point_parameter|Any floating point number encoded by heu, will multiply a scale and take the round, scale = 2 ** fixed_point_parameter. larger value may mean more numerical accuracy, but too large will lead to overflow problem.|Integer|N|Default: 20. Range: [1, 100].|
|first_tree_with_label_holder_feature|Whether to train the first tree with label holder's own features.|Boolean|N|Default: False.|
|batch_encoding_enabled|If use batch encoding optimization.|Boolean|N|Default: True.|
|enable_quantization|Whether enable quantization of g and h.|Boolean|N|Default: False.|
|quantization_scale|Scale the sum of g to the specified value.|Float|N|Default: 10000.0. Range: [0.0, 10000000.0].|
|max_leaf|Maximum leaf of a tree. Only effective if train leaf wise.|Integer|N|Default: 15. Range: [1, 32768].|
|rowsample_by_tree|Row sub sample ratio of the training instances.|Float|N|Default: 1.0. Range: (0.0, 1.0].|
|enable_goss|Whether to enable GOSS.|Boolean|N|Default: False.|
|top_rate|GOSS-specific parameter. The fraction of large gradients to sample.|Float|N|Default: 0.3. Range: (0.0, 1.0].|
|bottom_rate|GOSS-specific parameter. The fraction of small gradients to sample.|Float|N|Default: 0.5. Range: (0.0, 1.0].|
|early_stop_criterion_g_abs_sum|If sum(abs(g)) is lower than or equal to this threshold, training will stop.|Float|N|Default: 0.0. Range: [0.0, $\infty$).|
|early_stop_criterion_g_abs_sum_change_ratio|If absolute g sum change ratio is lower than or equal to this threshold, training will stop.|Float|N|Default: 0.0. Range: [0.0, 1.0].|
|tree_growing_method|How to grow tree?|String|N|Default: level.|

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|train_dataset|Input vertical table.|['sf.table.vertical_table']||

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_model|Output model.|['sf.model.sgb']||

### ss_glm_train


Component version: 0.0.1

generalized linear model (GLM) is a flexible generalization of ordinary linear regression.
The GLM generalizes linear regression by allowing the linear model to be related to the response
variable via a link function and by allowing the magnitude of the variance of each measurement to
be a function of its predicted value.
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|epochs|The number of complete pass through the training data.|Integer|N|Default: 10. Range: [1, $\infty$).|
|learning_rate|The step size at each iteration in one iteration.|Float|N|Default: 0.1. Range: (0.0, $\infty$).|
|batch_size|The number of training examples utilized in one iteration.|Integer|N|Default: 1024. Range: (0, $\infty$).|
|link_type|link function type|String|Y|Default: . Allowed: ['Logit', 'Log', 'Reciprocal', 'Identity'].|
|label_dist_type|label distribution type|String|Y|Default: . Allowed: ['Bernoulli', 'Poisson', 'Gamma', 'Tweedie'].|
|tweedie_power|Tweedie distribution power parameter|Float|N|Default: 1.0. Range: [0.0, 2.0].|
|dist_scale|A guess value for distribution's scale|Float|N|Default: 1.0. Range: [1.0, $\infty$).|
|eps|If the change rate of weights is less than this threshold, the model is considered to be converged, and the training stops early. 0 to disable.|Float|N|Default: 0.0001. Range: [0.0, $\infty$).|
|iter_start_irls|run a few rounds of IRLS training as the initialization of w, 0 disable|Integer|N|Default: 0. Range: [0, $\infty$).|
|decay_epoch|decay learning interval|Integer|N|Default: 0. Range: [0, $\infty$).|
|decay_rate|decay learning rate|Float|N|Default: 0.0. Range: [0.0, 1.0).|
|optimizer|which optimizer to use: IRLS(Iteratively Reweighted Least Squares) or SGD(Stochastic Gradient Descent)|String|Y|Default: . Allowed: ['SGD', 'IRLS'].|
|offset_col|Specify a column to use as the offset|String|N|Default: .|
|weight_col|Specify a column to use for the observation weights|String|N|Default: .|

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|train_dataset|Input vertical table.|['sf.table.vertical_table']||

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_model|Output model.|['sf.model.ss_glm']||

### ss_sgd_train


Component version: 0.0.1

Train both linear and logistic regression
linear models for vertical partitioning dataset with mini batch SGD training solver by using secret sharing.
- SS-SGD is short for secret sharing SGD training.
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
|eps|If the change rate of weights is less than this threshold, the model is considered to be converged, and the training stops early. 0 to disable.|Float|N|Default: 0.001. Range: [0.0, $\infty$).|

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|train_dataset|Input vertical table.|['sf.table.vertical_table']||

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|output_model|Output model.|['sf.model.ss_sgd']||

### ss_xgb_train


Component version: 0.0.1

This method provides both classification and regression tree boosting (also known as GBDT, GBM)
for vertical partitioning dataset setting by using secret sharing.
- SS-XGB is short for secret sharing XGB.
- More details: https://arxiv.org/pdf/2005.08479.pdf
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
|train_dataset|Input vertical table.|['sf.table.vertical_table']||

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
|in_ds|Input vertical table.|['sf.table.vertical_table']|Extra table attributes.(0) drop_features - Features to drop. |

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|out_ds|Output vertical table.|['sf.table.vertical_table']||

### psi


Component version: 0.0.1

PSI between two parties.
#### Attrs
  

|Name|Description|Type|Required|Notes|
| :--- | :--- | :--- | :--- | :--- |
|protocol|PSI protocol.|String|N|Default: ECDH_PSI_2PC. Allowed: ['ECDH_PSI_2PC', 'KKRT_PSI_2PC', 'BC22_PSI_2PC'].|
|sort|Sort the output.|Boolean|N|Default: False.|
|bucket_size|Specify the hash bucket size used in PSI. Larger values consume more memory.|Integer|N|Default: 1048576. Range: (0, $\infty$).|
|ecdh_curve_type|Curve type for ECDH PSI.|String|N|Default: CURVE_FOURQ. Allowed: ['CURVE_25519', 'CURVE_FOURQ', 'CURVE_SM2', 'CURVE_SECP256K1'].|

#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|receiver_input|Individual table for receiver|['sf.table.individual']|Extra table attributes.(0) key - Column(s) used to join. If not provided, ids of the dataset will be used. |
|sender_input|Individual table for sender|['sf.table.individual']|Extra table attributes.(0) key - Column(s) used to join. If not provided, ids of the dataset will be used. |

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|psi_output|Output vertical table|['sf.table.vertical_table']||

### train_test_split


Component version: 0.0.1

Split datasets into random train and test subsets.
- Please check: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
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
|input_data|Input vertical table.|['sf.table.vertical_table']||

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|train|Output train dataset.|['sf.table.vertical_table']||
|test|Output test dataset.|['sf.table.vertical_table']||

## stats

### ss_pearsonr


Component version: 0.0.1

Calculate Pearson's product-moment correlation coefficient for vertical partitioning dataset
by using secret sharing.
- For large dataset(large than 10w samples & 200 features), recommend to use [Ring size: 128, Fxp: 40] options for SPU device.
#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_data|Input vertical table.|['sf.table.vertical_table']|Extra table attributes.(0) feature_selects - Specify which features to calculate correlation coefficient with. If empty, all features will be used |

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|report|Output Pearson's product-moment correlation coefficient report.|['sf.report']||

### ss_vif


Component version: 0.0.1

Calculate Variance Inflation Factor(VIF) for vertical partitioning dataset
by using secret sharing.
- For large dataset(large than 10w samples & 200 features), recommend to use [Ring size: 128, Fxp: 40] options for SPU device.
#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_data|Input vertical table.|['sf.table.vertical_table']|Extra table attributes.(0) feature_selects - Specify which features to calculate VIF with. If empty, all features will be used. |

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|report|Output Variance Inflation Factor(VIF) report.|['sf.report']||

### table_statistics


Component version: 0.0.1

Get a table of statistics,
including each column's
1. datatype
2. total_count
3. count
4. count_na
5. min
6. max
7. var
8. std
9. sem
10. skewness
11. kurtosis
12. q1
13. q2
14. q3
15. moment_2
16. moment_3
17. moment_4
18. central_moment_2
19. central_moment_3
20. central_moment_4
21. sum
22. sum_2
23. sum_3
24. sum_4
- moment_2 means E[X^2].
- central_moment_2 means E[(X - mean(X))^2].
- sum_2 means sum(X^2).
#### Inputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|input_data|Input table.|['sf.table.vertical_table', 'sf.table.individual']||

#### Outputs
  

|Name|Description|Type(s)|Notes|
| :--- | :--- | :--- | :--- |
|report|Output table statistics report.|['sf.report']||
