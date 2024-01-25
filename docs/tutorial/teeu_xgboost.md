# TEEU Example: XGBoost

**Tips**

Before reading this article, it is strongly recommended to read [TEEU Getting Started Guide](./teeu.md) at first.

---

TEEU (`TEE` processing `U`nit) is a TEE device in SecretFlow. Through TEEU, users can conveniently put data in TEE for calculation, and achieve the purpose of protecting data integrity and security. 

This article will demonstrate how to run XGBoost in TEEU for model training.

## 1.1 Simulation mode

To facilitate users who do not have access to a real TEE environment, SecretFlow offers a TEEU simulation mode. This feature allows users to try out TEEU functions on an ordinary machine.
Code writing and usage in the simulation mode are almost same with the non-simulation mode, so it is recommended to use the simulation mode for quick experimental verification first.

Note that since the real TEE environment is not used, the simulation mode lacks security features that depend on the TEE environment, such as remote attestation and memory encryption isolation, and cannot protect data integrity and confidentiality. Simulation mode is not secure and should not be used in production, keep this in mind.

### Pre-work

#### Understand the SecretFlow deployment of multi-ray cluster mode

For security reasons, Ray running in TEE is an independent cluster, so currently SecretFlow only supports the use of TEEU in multiple Ray cluster mode. You can read the [SecretFlow Deployment Documentation](../getting_started/deployment.md#production) in advance to understand the deployment of multiple Ray clusters.

#### Prepare to run the simulated TEEU machine

At present, SecretFlow TEEU only provides docker images. Due to some limitations of the basic technology, TEE programs currently require a large amount of memory to run successfully. You need to ensure that the available memory for the Docker container is at least 30GB or more, depending on the size of the data to be processed in TEEU.

#### Deploy AuthManager

AuthManager is the module responsible for authorization management.

1. Download the docker image
```shell
docker pull secretflow/authmanager-release-sim-ubuntu:latest
```

2. Enter the docker image
```shell
docker run -it --net host secretflow/authmanager-release-sim-ubuntu:latest
```

3. (Optional) Configure TLS

AuthManager enables TLS by default. If you only use it for local simulation, you can turn off TLS by set `enable_tls` to `false` in `/root/occlum_release/config.yaml`.

4. Start the service

```shell
cd occlum_release
occlum run /bin/auth-manager --config_path /host/config.yaml
```
The default port is 8835. Feel free to modify the `port` in config.yaml if port conflicts.

### Example: XGBoost in TEEU

Next, we will demonstrate how to combine data from multiple parties in TEEU, and then use XGBoost to train it.

#### Example code

Assuming that Alice and Bob have the same feature space, but the sample space does not overlap with each other, and each has some user features, Alice and Bob intend to use TEEU to safely combine their samples and use XGBoost to train a model. At the same time, Carol acts as the provider of TEEU.

The core code of the above case is as follows.

```python
import secretflow as sf
import numpy as np

def xgb_demo(x_slices, y_slices):
    """
    Cancat the input x and y, then train them with XGBoost.
    """
    from sklearn.model_selection import train_test_split
    import xgboost as xgb

    # Concat x and y firstly.
    x = np.concatenate(x_slices)
    y = np.concatenate(y_slices)

    x_train, x_test = train_test_split(x, random_state=0)
    y_train, y_test = train_test_split(y, random_state=0)
    dtrain = xgb.DMatrix(data=x_train, label=y_train)
    dtest = xgb.DMatrix(data=x_test, label=y_test)
    num_boost_round = 16
    watchlist = [(dtest, "eval"), (dtrain, "train")]

    booster = xgb.train(
        {'num_parallel_tree': 4, 'subsample': 0.7, 'num_class': 3},
        num_boost_round=num_boost_round,
        dtrain=dtrain,
        evals=watchlist,
    )

    preds = booster.predict(dtest)
    labels = dtest.get_label()
    error = sum(
        1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]
    ) / float(len(preds))

    # `/host` in TEEU is mapped to the /root/occlum_instance in the docker container.
    booster.save_model('/host/model.json')

    return error


def gen_data():
    """
    Generate random classified data for simulation.
    """
    from sklearn.datasets import make_classification

    num_classes = 3
    x, y = make_classification(n_samples=1000, n_informative=5,
                           n_classes=num_classes)
    return x, y

# Alice generates its samples.
x_a, y_a = alice(gen_data, num_returns=2)()
# Bob generates its samples.
x_b, y_b = alice(gen_data, num_returns=2)()

from secretflow.device import TEEU

# mrenclave can be omitted in simulation mode.
teeu = TEEU('carol', mr_enclave='')

# Transfer data to teeu.
x_a_teeu = x_a.to(teeu, allow_funcs=xgb_demo)
y_a_teeu = y_a.to(teeu, allow_funcs=xgb_demo)

x_b_teeu = x_b.to(teeu, allow_funcs=xgb_demo)
y_b_teeu = y_b.to(teeu, allow_funcs=xgb_demo)

# Run xgb_demo.
res = teeu(xgb_demo)([x_a_teeu, x_b_teeu], [y_a_teeu, y_b_teeu])
error = sf.reveal(res)
print(f'Train error: {error}')

```

#### Alice runs the code

1. Start the ray master node

You should modify the following command to match the actual situation, as it currently assumes that Alice's Ray master node is listening at 192.168.0.10:10000.

```bash
ray start --head --node-ip-address="192.168.0.10" --port="10000" --include-dashboard=False --disable-usage-stats
```

2. Generate a public-private key pair

As Alice's data needs to be encrypted and sent to TEEU, it is imperative to generate a pair of public and private keys. Below, you may find the code that, upon execution, generates the public and private keys, which will be stored in the current directory in PEM format as "private_key.pem" and "public_key.pem", respectively.

```bash
openssl genrsa -3 -out private_key.pem 3072
openssl rsa -in private_key.pem -pubout -out public_key.pem
```

3. Execute code

Add the SecretFlow initialization related code in front of the code to get the following code.
First, you need to modify the configuration items in the code.
- The code assumes that Alice's communication address is 192.168.0.10:20001, please modify it according to the actual situation
- You need to fill in the correct `auth_manager_config`
  - `host` is the listening address of the AuthManager service
  - `ca_cert` is the CA certificate address of AuthManager, if AuthManager does not start with TLS, no configuration is required.

Suppose we save the code as `demo.py`, and then execute `python demo.py` on Alice's machine.

```python
import secretflow as sf

cluster_config = {
    'parties': {
        'alice': {'address': '192.168.0.10:20001', 'listen_address': '0.0.0.0:20001'},
        'bob': {'address': '192.168.0.20:20001', 'listen_address': '0.0.0.0:20001'},
        'carol': {'address': '192.168.0.30:20001', 'listen_address': '0.0.0.0:20001'},
    },
    'self_party': 'alice',
}

party_key_pair = {
    'alice': {'private_key': './private_key.pem', 'public_key': './public_key.pem'}
}

auth_manager_config = {
    'host': 'host of AuthManager',
    'ca_cert': 'path_of_AuthManager_ca_certificate',
    'mr_enclave': ''
}

# Connect to alice's ray
sf.init(
    address='192.168.0.10:10000',
    cluster_config=cluster_config,
    party_key_pair=party_key_pair,
    auth_manager_config=auth_manager_config,
    tee_simulation=True,
)

import numpy as np

def xgb_demo(x_slices, y_slices):
    """
    Cancat the input x and y, then train them with XGBoost.
    """
    from sklearn.model_selection import train_test_split
    import xgboost as xgb

    # Concat x and y firstly.
    x = np.concatenate(x_slices)
    y = np.concatenate(y_slices)

    x_train, x_test = train_test_split(x, random_state=0)
    y_train, y_test = train_test_split(y, random_state=0)
    dtrain = xgb.DMatrix(data=x_train, label=y_train)
    dtest = xgb.DMatrix(data=x_test, label=y_test)
    num_boost_round = 16
    watchlist = [(dtest, "eval"), (dtrain, "train")]

    booster = xgb.train(
        {'num_parallel_tree': 4, 'subsample': 0.7, 'num_class': 3},
        num_boost_round=num_boost_round,
        dtrain=dtrain,
        evals=watchlist,
    )

    preds = booster.predict(dtest)
    labels = dtest.get_label()
    error = sum(
        1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]
    ) / float(len(preds))

    # `/host` in TEEU is mapped to the /root/occlum_instance in the docker container.
    booster.save_model('/host/model.json')

    return error


def gen_data():
    """
    Generate random classified data for simulation.
    """
    from sklearn.datasets import make_classification

    num_classes = 3
    x, y = make_classification(n_samples=1000, n_informative=5,
                           n_classes=num_classes)
    return x, y

# Alice generates its samples.
x_a, y_a = alice(gen_data, num_returns=2)()
# Bob generates its samples.
x_b, y_b = alice(gen_data, num_returns=2)()

from secretflow.device import TEEU

teeu = TEEU('carol', mr_enclave='')

# Transfer data to teeu.
x_a_teeu = x_a.to(teeu, allow_funcs=xgb_demo)
y_a_teeu = y_a.to(teeu, allow_funcs=xgb_demo)

x_b_teeu = x_b.to(teeu, allow_funcs=xgb_demo)
y_b_teeu = y_b.to(teeu, allow_funcs=xgb_demo)

# Run xgb_demo.
res = teeu(xgb_demo)([x_a_teeu, x_b_teeu], [y_a_teeu, y_b_teeu])
error = sf.reveal(res)
print(f'Train error: {error}')

```

#### Bob runs the code

1. Start the ray master node

You should modify the following command to match the actual situation, as it currently assumes that Bob's Ray master node is listening at 192.168.0.20:10000.
```bash
ray start --head --node-ip-address="192.168.0.20" --port="100000" --include-dashboard=False --disable-usage-stats
```

2. Generate a public-private key pair

As Bob's data needs to be encrypted and sent to TEEU, it is imperative to generate a pair of public and private keys. Below, you may find the code that, upon execution, generates the public and private keys, which will be stored in the current directory in PEM format as "private_key.pem" and "public_key.pem", respectively.
```bash
openssl genrsa -3 -out private_key.pem 3072
openssl rsa -in private_key.pem -pubout -out public_key.pem
```

3. Run the code

Similar to Alice, add the SecretFlow initialization code in front of the code to get the following code.
First, you need to modify the configuration items in the code.

- The code assumes that Bob's communication address is 192.168.0.20:20001, please modify it according to the actual situation
- You need to fill in the correct `auth_manager_config`
- `host` is the listening address of the AuthManager service
- `ca_cert` is the CA certificate address of AuthManager, if AuthManager does not start tls, no configuration is required.

Suppose we save the code as `demo.py`, and then execute `python demo.py` on Bob's machine.

```python
import secretflow as sf

cluster_config = {
    'parties': {
        'alice': {'address': '192.168.0.10:20001', 'listen_address': '0.0.0.0:20001'},
        'bob': {'address': '192.168.0.20:20001', 'listen_address': '0.0.0.0:20001'},
        'carol': {'address': '192.168.0.30:20001', 'listen_address': '0.0.0.0:20001'},
    },
    'self_party': 'bob',
}

party_key_pair = {
    'bob': {'private_key': './private_key.pem', 'public_key': './public_key.pem'}
}

auth_manager_config = {
    'host': 'host of AuthManager',
    'ca_cert': 'path_of_AuthManager_ca_certificate',
    'mr_enclave': ''
}

# Connect to Bob's ray
sf.init(
    address='192.168.0.20:10000',
    cluster_config=cluster_config,
    party_key_pair=party_key_pair,
    auth_manager_config=auth_manager_config,
    tee_simulation=True,
)

import numpy as np

def xgb_demo(x_slices, y_slices):
    """
    Cancat the input x and y, then train them with XGBoost.
    """
    from sklearn.model_selection import train_test_split
    import xgboost as xgb

    # Concat x and y firstly.
    x = np.concatenate(x_slices)
    y = np.concatenate(y_slices)

    x_train, x_test = train_test_split(x, random_state=0)
    y_train, y_test = train_test_split(y, random_state=0)
    dtrain = xgb.DMatrix(data=x_train, label=y_train)
    dtest = xgb.DMatrix(data=x_test, label=y_test)
    num_boost_round = 16
    watchlist = [(dtest, "eval"), (dtrain, "train")]

    booster = xgb.train(
        {'num_parallel_tree': 4, 'subsample': 0.7, 'num_class': 3},
        num_boost_round=num_boost_round,
        dtrain=dtrain,
        evals=watchlist,
    )

    preds = booster.predict(dtest)
    labels = dtest.get_label()
    error = sum(
        1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]
    ) / float(len(preds))

    # `/host` in TEEU is mapped to the /root/occlum_instance in the docker container.
    booster.save_model('/host/model.json')

    return error


def gen_data():
    """
    Generate random classified data for simulation.
    """
    from sklearn.datasets import make_classification

    num_classes = 3
    x, y = make_classification(n_samples=1000, n_informative=5,
                           n_classes=num_classes)
    return x, y

# Alice generates its samples.
x_a, y_a = alice(gen_data, num_returns=2)()
# Bob generates its samples.
x_b, y_b = alice(gen_data, num_returns=2)()

from secretflow.device import TEEU

teeu = TEEU('carol', mr_enclave='')

# Transfer data to teeu.
x_a_teeu = x_a.to(teeu, allow_funcs=xgb_demo)
y_a_teeu = y_a.to(teeu, allow_funcs=xgb_demo)

x_b_teeu = x_b.to(teeu, allow_funcs=xgb_demo)
y_b_teeu = y_b.to(teeu, allow_funcs=xgb_demo)

# Run xgb_demo.
res = teeu(xgb_demo)([x_a_teeu, x_b_teeu], [y_a_teeu, y_b_teeu])
error = sf.reveal(res)
print(f'Train error: {error}')

```

#### Carol runs code (executed in TEE)

Run the SecretFlow TEE image firstly.

```bash
docker run -it --network host secretflow/secretflow-teeu:latest
```

Similarly, add the SecretFlow initialization code in front of the code to get the following code. Unlike the previous one, Carol's code needs to run in TEE, so some extra steps are required.
First, you need to modify the configuration items in the code.

1. In the code, it is assumed that Carol's communication address is 192.168.0.30:20001, please modify it according to the actual situation
2. You need to fill in the correct `auth_manager_config`
  - `host` is the listen address of AuthManager
  - `ca_cert` is the CA certificate path of AuthManager, if AuthManager does not enable TLS, no configuration is required.

After modification, please save the file to `/root/occlum_instance/image/root/demo.py`.

```python

# Generate tls cert and key at first.
from tls_cert import generate_self_signed_tls_certs

generate_self_signed_tls_certs('/root/server.crt', '/root/server.key')


import secretflow as sf

cluster_config = {
    'parties': {
        'alice': {'address': '192.168.0.10:20001', 'listen_address': '0.0.0.0:20001'},
        'bob': {'address': '192.168.0.20:20001', 'listen_address': '0.0.0.0:20001'},
        'carol': {'address': '192.168.0.30:20001', 'listen_address': '0.0.0.0:20001'},
    },
    'self_party': 'carol',
}

auth_manager_config = {
    'host': 'host of AuthManager',
    'ca_cert': 'path_of_AuthManager_ca_certificate',
    'mr_enclave': ''
}

# Start a local Ray.
sf.init(
    address='local',
    cluster_config=cluster_config,
    auth_manager_config=auth_manager_config,
    tee_simulation=True,
    _temp_dir="/host/tmp/ray",
    _plasma_directory="/tmp",
)

import numpy as np

def xgb_demo(x_slices, y_slices):
    """
    Cancat the input x and y, then train them with XGBoost.
    """
    from sklearn.model_selection import train_test_split
    import xgboost as xgb

    # Concat x and y firstly.
    x = np.concatenate(x_slices)
    y = np.concatenate(y_slices)

    x_train, x_test = train_test_split(x, random_state=0)
    y_train, y_test = train_test_split(y, random_state=0)
    dtrain = xgb.DMatrix(data=x_train, label=y_train)
    dtest = xgb.DMatrix(data=x_test, label=y_test)
    num_boost_round = 16
    watchlist = [(dtest, "eval"), (dtrain, "train")]

    booster = xgb.train(
        {'num_parallel_tree': 4, 'subsample': 0.7, 'num_class': 3},
        num_boost_round=num_boost_round,
        dtrain=dtrain,
        evals=watchlist,
    )

    preds = booster.predict(dtest)
    labels = dtest.get_label()
    error = sum(
        1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]
    ) / float(len(preds))

    # `/host` in TEEU is mapped to the /root/occlum_instance in the docker container.
    booster.save_model('/host/model.json')

    return error


def gen_data():
    """
    Generate random classified data for simulation.
    """
    from sklearn.datasets import make_classification

    num_classes = 3
    x, y = make_classification(n_samples=1000, n_informative=5,
                           n_classes=num_classes)
    return x, y

# Alice generates its samples.
x_a, y_a = alice(gen_data, num_returns=2)()
# Bob generates its samples.
x_b, y_b = alice(gen_data, num_returns=2)()

from secretflow.device import TEEU

teeu = TEEU('carol', mr_enclave='')

# Transfer data to teeu.
x_a_teeu = x_a.to(teeu, allow_funcs=xgb_demo)
y_a_teeu = y_a.to(teeu, allow_funcs=xgb_demo)

x_b_teeu = x_b.to(teeu, allow_funcs=xgb_demo)
y_b_teeu = y_b.to(teeu, allow_funcs=xgb_demo)

# Run xgb_demo.
res = teeu(xgb_demo)([x_a_teeu, x_b_teeu], [y_a_teeu, y_b_teeu])
error = sf.reveal(res)
print(f'Train error: {error}')

```

Then we run the script with the following command.

```bash
cd /root/occlum_instance
openssl genrsa -3 -out private_key.pem 3072
openssl rsa -in private_key.pem -pubout -out public_key.pem
occlum build --sgx-mode sim --sign-key private_key.pem
occlum run /bin/python3.8 /root/demo.py
```

You can check model file at `/root/occlum_instance/model.json` when finished.

## 1.2 Non-simulation mode

When it is necessary to use the real TEE environment to protect the confidentiality and integrity of the data in the computing process, the user needs to enable the non-simulation mode, and at this time, the security mechanisms provided by the TEE such as remote attestation and memory encryption will be enabled. To enable the non-simulation mode, the user needs to have the TEE hardware supported by the current SecretFlow TEEU. Currently, SecretFlow only supports Intel SGX2.0, and more TEE types will be supported in the future.

Please check [Non-simulation](./teeu.md#summary) for running in non-simulation mode.
