# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Types of changes
`Added ` for new features.
`Changed` for changes in existing functionality.
`Deprecated` for soon-to-be removed features.
`Removed` for now removed features.
`Fixed` for any bug fixes.
`Security` in case of vulnerabilities.

## [v1.4.0.dev240205] - 2024-02-05
### Added
- Add Criteo(100M) Dataset.
- Add split learning nn component.
- Add sgb & ss_xgb model export.
- Split learning support prediction for file input.
- GPU support for split learning attack algorithm.

### Changed
- Unify nn Callback naming.

## [v1.4.0.dev240129] - 2024-01-29
### Added
- Split learning in PyTorch supports GPU in debug mode.
- Split learning in PyTorch supports pipeline strategy.
- Sparse compression supports the compression of multi-dimensional data, such as images.
- SLModel supports input in the form of file names and allows for multiple label inputs.

- Component: prediction operator results can now save feature columns.
- Component: binning operators can now view reports directly.
- Component: binning rule modification is supported.
- Component: support model exports, support both online and offline prediction.

### Fixed
- fix: bump rayfed to 0.2.0a18 to fix shutdown lock.
- Component: improve GLM training accuracy.
- Component: reduce GLM training memory and computation cost.

## [v1.4.0.dev240122] - 2024-01-22
### Added
- Add a new feature of chunked computation to control the memory peak.
- The binning component supports displaying binning rules.
-
### Fixed
- Fix a bug that occurs when saving the SGB model fails when the number of trees in the model is 1.
- Fix bin rule report.

## [1.4.0.dev240116] - 2024-01-16
### Added
- sgb/ss_xgb/ss_glm/ss_sgd can save specified columns with result now.
- ss_glm: expose two params infeed size and newton.
- SplitLearning
  - Add a defense method against data reconstruction attacks.
  - Adapt replay attack and gradreplace attack for deepfm.
  - Add expoit attack.

### Changed
- Component: update fillna descriptions.

### Fixed
- Fix SS-GLM toturial.

## [1.4.0.dev240103] - 2024-1-3
### Added
- Add grad replace attack and replay attack.
- Add autoattack benchmark examples.
- Expose job_name param in sf.init.

### Changed
- Bump rayfed version: optimizing Error Propagation and Capture.
- Component: woe_bins requires at least 5 bins to read.
- Component: add barrier_on_shutdown as sf cluster config.

## [1.4.0.dev231225] - 2023-12-25
### Added
- Add DataProxy binary writer.
- Add PSI v2 APIs.
- Add sl_deepfm_torch, sl_dnn_tf and sl_dnn_torch applications.
- Support sgb conversion into a single model.
- Support debug mode in tuner.
- Docs: Update installation and deployment docs.
- Docs: Add fl gpu docs.
- Docs: Add debug mode docs.

### Changed
- Component: onehot_encode component limits size of col_rules.
- Component: sgb_predict component rm batch_size param.

### Fixed
- Fix pytorch_audio_classification example and tutorial.
- Fix bin substitution for incomplete rule.
- Fix sgb_predict component when all features come from a party.

## [1.3.0.dev231219] - 2023-12-19
### Added
- Make barrier_on_shutdown optional.
- Support SGB label holder without features.
- Support SL Model training on file data with mutiple labels.
- Add SL ResNet and VGG application.
- Secretflow ic: Add package interconnection protobuf files.
- Component: Add feature calculate component to generate new features by performing calculations on original features.
- Component: Support SGB prediction on big dataset.

### Changed
- SGB optimize memory usage in prediction.
- Component: Bump groupby statistics version.
- Component: Improve translation.

### Fixed
- Component: Fix woe io and fillna.


## [1.3.0.dev231211] - 2023-12-11
### Added
- Add IO component including read, write and identity.
- Change groupby component to by-query style.


## [1.3.0.dev231128] - 2023-11-28

### Added
- Add secretflow tuner for automl and autoattack.
- Add IO component including read, write and identity.
- Change groupby component to by-query style.

## [1.3.0.dev231120] - 2023-11-20

### Added

- Support file data input in SLModel.
- Expose copts in SPU devices.
- Component: Add benchmarks.
- Add federated callback framework.
- Add new agg method -- concat,sum.
- Add ic_mode.

### Changed

- Component: add upper_bound for max_group_size in groupby_statistics.
- Component: modify test_size and train_size restrictions in train_test_split.
- Clear legacy history class in FL.

### Fixed

- Kuscia adapter: check datasource only when author matches.


## [1.3.0.dev231109] - 2023-11-9

### Added

- Component: Support eq_range binning.
- Component: Support TLS in nsjail.
- Component: Add test framework.
- Component: Adapt to DataProxy.

### Changed

- Component: Select features in binning.
- Component: biclassification_eval return nan values if min_item_cnt_per_bucket doesn't match.

### Fixed

- Enhance sgb.
- Component: Fix groupby_statistics

## [1.2.0] - 2023-11-8

### Added

- Add debug mode.
- Support GLM model transition from MPC version to Federated version.
- FLModel supports PFL to allow custom aggregation logic on server.
- Add split learning applications: BST, MMoE.
- Add sparse, quantized and mixed compressor.
- Add polars backend for dataframe and SL model to enhance the data processing performance.

### Changed

- Refactor data preprocessing module (VDataFrame, Partition).
- Doc: rearrange docs for split recommendation suite: SplitRec.
- Doc: Update PSI benchmark.
- Component: expose cross_silo_comm_backend option for secretflow init.
- Component: add vert_binning.
- Component: migrate to [SecretFlow Open Specification](https://github.com/secretflow/spec).

## [1.2.0.dev231025] - 2023-10-25
### Added
- Support naive sl on torch backend.
- Add kwargs for custom strategy in sl.
- Add bst and mmoe in sl applications.

### Changed
- Input attributes of some components are modified.
- Convert domain data to individual tables in preprocess_sf_node_eval_param of kuscia adapter.

## [1.2.0.dev231016] - 2023-10-16
### Added
- Secretflow support debug mode.
- Add vert binning for equal range bining method

### Changed
- The data preprocessing module (VDataFrame, Partition) has been refactored, enhancing the data processing performance (primarily targeting the Polars backend).

### Fixed
- Fix error when flmodel with tf backend use gpu.
- Fix kuscia adapter

## [1.2.0.dev231009] - 2023-10-9

### Added
- FLModel supports PFL to allow custom aggregation logic on server.

## [1.2.0.dev230926] - 2023-9-26

### Changed
- component: expose cross_silo_comm_backend option for secretflow init.

### Fixed
- SPU device: rm invalid use_link option.

## [1.2.0.dev230925] - 2023-9-25

### Changed
- update psi benchmark.

### Fixed
- fix tls on kuscia.

## [1.2.0.dev230918] - 2023-9-18

### Added
- support GLM model transition from MPC version to Federated version.
- add sparse, quantized and mixed compressor.
- add polars backend for dataframe and SL model.
- add DisPFL adn VFGNN example.
- add cross_silo_comm_backend option to SFClusterConfig.RayFedConfig.

### Fixed
- fix dataset build in SL.

## [1.1.0b0] - 2023-9-6

### Added
- SLModel: supports quantization compression algorithm, reducing communication volume by 2-4 times.
- SLModel: supports pipeline strategy, which can accelerate model training by 2-4 times in most scenarios.
- SLModel: PyTorch backend supports GPU.
- SLModel: introduces two attack and defense algorithms, LIA and FIA, for testing model security in the research and development stage.
- SLModel: supports a mode where one party only provides labels without providing features.
- Component: GLM train and predict components.
- Support the usage of brpc link as a backend for cross-silo communication.

### Changed
- Component: add more parameters for SGB components.

## [1.1.0.dev230825] - 2023-8-25

### Added
- Stateful task for teeu
- docs: DeepFM translation

### Changed
- Switch to shared workflow

## [1.1.0.dev230818] - 2023-8-18

### Added
- Add five papers in Vertical Federated Learning
- docs: update references on homomorphic encryption
- Add new quantized compressor method and tutorial

## [1.1.0.dev230811] - 2023-8-11

### Fixed
- PSI use psi_csv in psi comp.

### Added
- GLM train and predict components
## [1.1.0b0.dev1] - 2023-8-7

### Added
- Support the usage of brpc link as a backend for cross-silo communication.

## [1.1.0b0.dev0] - 2023-8-7

### Added
- Label inference attack v3

## [1.0.0b2] - 2023-7-28
### Changed
- SGB upgrade: use SGBFactory to replace SGB, update parameters and tutorials. SGB now supports more functionalities.
- Predict supports callbacks, call the callback function before/after prediction starts and after every step.
## [1.0.0a1] - 2023-7-26
### Fixed
- SLModel fix bug in handling data with databuilder
- The FLModel solves the problem of the production mode hanging due to a small batch size.

## [1.0.0a0] - 2023-7-3
### Added
- SLModel supports AggLayer
- SLModel（nn/deepfm）supports one party providing features and the other party providing labels.
- Component Specification and SecretFlow Component List v0.0.1.

### Changed
- Bump spu to 0.4.1b0

### Fixed
- Fix logic error when sl base model load from none

## [0.8.3b0] - 2023-6-14
### Added
- Split learning add application of deepfm for recommendation scenarios.
- Split learning add pytorch support.
- Add IO tutorials for federated learning.
- Reorg DP strategies
- SGB and XGB refactor. Add data checks. Improve qcut.
- Add the preview version of components.

### Changed
- Bump spu to 0.3.3b2

### Fixed
- TEEU function serialization.

## [0.8.2b3] - 2023-5-15
### Fixed
- Correct TEEU function serialization protocol and docs.

## [0.8.2b2] - 2023-5-9
### Fixed
- Fix SPU compilation cache bug

### Changed
- Bump spu to 0.3.3b2

## [0.8.2b1] - 2023-4-27
### Fixed
- Add missing __init__ files.

## [0.8.2b0] - 2023-4-19
### Added
- TEE python Unit(TEEU) is introduced as the TEE cryptographic device and enables authorized computation with authorized data in TEE. TEEU brings more possibilities for hybrid computation.
- An experimental SecretFlow component design.
- SGB feature: add support for pre-pruning and model save & load.

### Changed
- Use pytest instead of unittest.
- Bump spu to 0.3.3b0
- Bump heu to 0.4.3b2

### Fix
- Fix hess lr auc err with large learning_rate.

## [0.8.1b0] - 2023-4-7
### Added
- SecureBoost and its benchmark.
- Unbalanced PSI
  - Two sub-protocols for generating cache and transmitting cache.
  - Online with shuffling sub-protocol, supporting big data parties to obtain data.
- Semi-homomorphic encryption protocol - OU

### Changed
- Bump spu to 0.3.2b12

### Fix
- Fix psi_join_csv output columns error.
- Fix heu object decode.

## [0.8.0b1] - 2023-3-31
- Bump spu to 0.3.2b11

## [0.8.0b0] - 2023-3-28

### Added
- SCQL (Secure Collaborative Query Language).

### Changed
- Bump spu to 0.3.2b9.


## [0.7.18b5] - 2023-3-22

### Changed
- Bump many dependencies for security fix.

## [0.7.18b4] - 2023-3-7

### Changed
- Bump RayFed to 0.1.1a.
### Fix
- give min num_cpus for simulation.

## [0.7.18b3] - 2023-2-13
### Fix
- add __init__.py to sl tensorflow strategy folder.

## [0.7.18b2] - 2023-2-9
### Fix
- add party as resource label in simulation mode.


## [0.7.18b1] - 2023-1-30
### Changed
- add an option whether exit on cross-silo sending.
- put all requires in one file except dev.

### Fix
- Fix psi_join_csv output columns error.
- fix heu object decode.

## [0.7.18] - 2023-01-09

### Changed
- Bump:
  - rayfed to 0.1.0b0.
  - spu to 0.3.1b9.
  - sf-heu to 0.3.2b1.

## [0.7.17] - 2022-12-21

### Added
- add export_model api for SLModel
- add get_params api for preprocessing

### Changed
- VDataFrame read_csv uses spu.psi_csv instead of spu.psi_df to reduce memory usage.
- Bump rayfed to 0.1.0a10.
- Control the concurrency of ray task in SLModel fit/predict/evaluate.

### Fixed
- psi join multi-key sort command parameters

## [0.7.16] - 2022-12-16

### Changed
- Optimize psi join memory usage.

## [0.7.15] - 2022-12-14

### Changed
- Bump rayfed to 0.1.0a9.

### Fixed
- SS GLM distribution type err.

## [0.7.14] - 2022-12-12
### Added
- GLM.

### Changed
- bump rayfed to 0.1.0a8.

### Fixed
- SSXgb for user_specified_num_returns=1.
- SLModel sets steps_per_epoch for worker.
- set num_cpu and resources only when start ray with local mode.

## [0.7.13] - 2022-12-08
### Added
- Multi controller based on rayfed.
### Changed
- Rewrite data builder.
- Bump to yacl.
- utils.testing.cluster_def supports NPC.

### Fixed
- FLModel save path.
- SSXgb col sub error.
- HomeXgb callback.
- Compress mask.
- spu.__call__ fails when SPUCompilerNumReturnsPolicy is FROM_USER and user_specified_num_returns is 1.

## [0.7.12] - 2022-11-18
### Changed
- OneHotEncoder remove PYUObject property.

### Fixed
- Add sl_model metrics wrap to fix format error.

## [0.7.11] - 2022-11-15
### Added
- Add Finetune and FedEval to SFXgboost
- Add SLModel support multi parties(>=2)

### Changed
- FLModel supports most metrics of regression and classification

### Fixed
- SLModel can be initialized without model.
- PSI doc typos.
- Fix the logic error bug of savemodel and loadmodel path in SLModel and FLModel

## [0.7.10] - 2022-10-25
### Added
- Add scorecard.
- Add replace/mode function to DataFrame.
- Add round function to VDataFrame.
- Add psi_join_csv and psi_join_df.
- Add preprocessing.LogroundTransformer.
- Add args to preprocessing.OneHotEncoder.

### Changed
- Bump dependencies
  - secretflow-ray to 2.0.0.dev2
- Update psi_df doc.
- Optimize sl_model by tf_funciton.
- Add curve parameter for ecdh psi.
- Protect biclassification, psi and pva with pyu object.
- Modify XgbModel predict api.

### Fixed
- Raise exception if spu_fe.compile fails.
- Fix quantile security vulnerability.
- Fix woe bin bugs.
- Fix psi_join recv timeout.

## [0.7.9] - 2022-10-24
### Added
- omp_num_threads param for secretflow init().
- Regression and biclassification evaluation.
- Xgboost evaluation.
- Horizontal fl supports default naive aggreagte for metrics.
- PVA calculation.

### Changed
- Remove graph util NodeDataLoader.

### Fixed
- VDataFrame docstring.
- Remove dependencies
  - dgl

### Changed
- Get rid of import tensorflow/torch when import secretflow.

## [0.7.8] - 2022-9-22
### Added
- Add license file

### Fixed
- Fix sl predict & remove reveal
- Fix typos in function docs.

### Changed
- Bump dependencies
  - TensorFlow to 2.10.0
  - Jax to 0.3.17
  - Jaxlib to 0.3.15

## [0.7.7] - 2022-9-16
### Changed
- Bump dependencies
  - sf-heu to 0.2.0
  - spu to 0.2.5

## [0.7.6] - 2022-09-08
### Fixed
- Missing requirements in dev-requirements.txt.

## [0.7.5] - 2022-09-05
### Added
- SPU config param: throttle_window_size

## [0.7.4] - 2022-09-01
### Added
- PSI param: bucket_size

### Changed
- SPU config param http_timeout_ms defaults to 120s.
- Bump dependencies
  - sf-heu to 0.1.3.2

## [0.7.3] - 2022-08-29
### Added
- Add pytorch backend for fl model for classification
- Add FL Dp strategy

### Changed
- Update document
- Shrink docker image size
- Bump dependencies
    - sf-heu to 0.1.3.1

## [0.7.2] - 2022-08-25
### Changed
- Use secretflow-ray instead of ray.

## [0.7.1] - 2022-08-25
### Added
- Add steps_per_epoch parameter to callback function of SLBaseTFModel.

### Changed
- Bump dependencies
  - sf-heu to 0.1.2
- Remove example of mixlr as mixlr is in official code already.

### Fixed
- Fix psi docs.

## [0.7.0] - 2022-08-23
### Added
- Pytorch backend and FL strategy.
- SS pvalue.
- Horizontal NN global DP with RDP accountant.
- HEU supports encrypt with audit log.

### Changed
- HEU uses c++ numpy api.
- Update ant pypi address.
- Complete cluster model deployment doc.
- Use multiprocess.cpu_count instead of multiprocessing.cpu_count for compatibility with macOS.

### Fixed
- Fix sl gnn test.

## [0.6.17] - 2022-08-11
### Fixed
- fix model handle_data parties_length by adding partition_shape to dataframe

## [0.6.16] - 2022-08-08
### Added
- SS VIF.
- FL strategy: FedProx.
- Split GNN.

### Changed
- Remove duplicated shape_spu_to_np & dtype_spu_to_np in spu.py.

## [0.6.15] - 2022-08-02
### Added
- Development and release docker.
- FL model strategy.
- Sigmoid approximation in python.
- SS LR.
- Verical FL LR.
- Auto ray.get for nested params with pyu objects in proxy decorated cls.
- Link desc in spu construction.

### Changed
- Refactor datasets from oss instead of lfs.
- Many doc improvements.

### Fixed
- SecureAggregator `average` when weights are multi-dimensions.

## [0.6.14] - 2022-07-07
### Added
- Vertical dp.

### Changed
- Many docs improvements.

### Fixed
- Increase H2A mask bits
- Include c++ lib in setup.

## [0.6.13] - 2022-06-30
### Added
- simulation.dataset for tutorial
- update tutorial of FL SL & SFXgboost
- add csv stream reader for FL

## [0.6.12] - 2022-06-29
## [0.6.11] - 2022-06-28
### Fixed
- fix sf.init argument.

## [0.6.10] - 2022-06-27
### Fixed
- typos, grammatical errors, implicit in docs.

## [0.6.9] - 2022-06-27
### Added
- Secretflow shutdown.

### Fixed
- LabelEncoder returns np.int dtype.

## [0.6.8] - 2022-06-22
### Added
- FlModel supports csv loader.

### Changed
- Rename PPU to SPU.

## [0.6.7] - 2022-06-16
### Added
- MixLR demo.
- DP on split learning DNN.
- XGBoost tutorial.
- DataFrame and FedNdarray support astype method.

### Changed
- Use lfs instead of http file.
- FL model requires model define no more when using load_model.
- dataframe.to_csv returns object ref.

### Security
- Use more secure random.
- Complete security and not-for-production warning.

## [0.6.6] - 2022-06-09
### Added
- SplitDNN dp.
- Use lfs.
- XGboost tutorial.

### Fixed
- DataFrame.to_csv returns object refs for further wait.

## [0.6.5] - 2022-06-06
### Added
- Runtime_config as input to utils.testing.cluster_def.
- SFXgboost optimization.
- Horizontal preprocessing
  - StardardScaler
  - KBinsDiscretizater
  - Binning
- Vertical preprocessing: WOE binning and substitution.
- HEU supports int/fxp data type.
- HEU Object supports slice and sum.
- Differential privacy.
- API docstrings.
- Custom pytree node to ppu.
- English docs.

### Changed
- Remove default reveal in aggreagtor and compartor.
- Bump dependencies
  - jax to 0.3.7
  - sf-heu to 0.0.5
  - sf-ppu to 0.0.10.4
- FL model: up early stop from step to epoch.
- SecureAggregation uses powers of 2.
- Rename vdf partitions_dimensions to partition_columns.
- Use *args instead of args in aggregation for reducing ray task dependency.

### Fixed
- FL model progress bug.
- train_test_split typo.



## [0.6.4]
- Fix PPU dtype mismatch caused by JAX 32bit mode.
- Vertical PearsonR.

## [0.6.3]
- FlModel/SlModel support model path dict.

## [0.6.2]
- Upgrade sf-ppu version to 0.0.7.1

## [0.6.1]
- More perfect HEU
- Split learning benchmark model
- SFXgboost for homo xgboost training

## [0.6.0]
- FL: different batch size for different clients.
- Wait method for pyu objects.
- FLModel evaluate returns detailed metrics.

## [0.0.6] - 2022-04-06
- PPU listen address.
