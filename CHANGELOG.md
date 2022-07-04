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
- FLModelTf evaluate returns detailed metrics.

## [0.0.6] - 2022-04-06
- PPU listen address.