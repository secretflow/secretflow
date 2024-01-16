# Repository layout

This is a high level overview of how the repository is laid out. Some major folders are listed below:

* [benchmark_examples/](benchmark_examples/): scripts for secretflow component benchmark.
* [docker/](docker/): scripts to build secretflow release and dev docker images.
* [docs/](docs/): documents written in reStructuredText, Markdown, Jupyter-notebook.
* [examples/](examples/): examples of secretflow.
* [secretflow/](secretflow/): the core library.
    * [component/](secretflow/component/): secretflow components.
    * [compute/](secretflow/compute/): wrapper for pyarrow compute functions.
    * [data/](secretflow/data/): horizontal, vertical and mixed DataFrame and Ndarray (like pandas and numpy).
    * [device/](secretflow/device/): various devices and their kernels, such as PYU, SPU, HEU, etc.
    * [distributed/](secretflow/distributed/): logics related to Ray and RayFed.
    * [ic/](secretflow/ic/): interconnection.
    * [kuscia/](secretflow/kuscia/): adapter to kuscia.
    * [ml/](secretflow/ml/): federated learning and split learning algorithms.
    * [preprocessing/](secretflow/preprocessing/): preprocessing functions.
    * [protos/](secretflow/protos/): Protocol Buffers messages.
    * [security/](secretflow/security/): privacy related algorithms, such as secure aggregation, differential privacy.
    * [spec/](secretflow/spec/): generated code of spec Protocol Buffers messages.
    * [stats/](secretflow/stats/): statistics functions.
    * [tune/](secretflow/tune/): functions related to tuners.
    * [utils/](secretflow/utils/): miscellaneous utility functions.
* [secretflow_lib/](secretflow_lib/): some core functions written in C++ and their Python bindings.
* [tests/](tests/): unit tests with pytest. 
