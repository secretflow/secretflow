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

## Staging
## [0.7.11] - 2022-11-15
### Added
- Add Finetune and FedEval to SFXgboost
- Add SLModel support multi parties(>=2)

### Changed
- FLModel support most metrics of regression and classification

### Fixed
- SLModel can be initialized without model.
- PSI doc typos.

## [0.7.10] - 2022-10-25
### Added
- Add score card.
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
- Auto ray.get for nested params with pyu objects in proxy decoreted cls.
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
- DataFrame and FedNdarray supports astype method.

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
- SecureBoost optimization.
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
- Fix PPU dtype mismatch causing by JAX 32bit mode.
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