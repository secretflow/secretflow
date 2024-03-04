# TEEU Example: OneHotEncoder

**Tips**

Before reading this article, it is strongly recommended to read [TEEU Getting Started Guide](../teeu.md) at first.

---

TEEU (`TEE` processing `U`nit) is a TEE device in SecretFlow. Through TEEU, users can conveniently put data in TEE for calculation, and achieve the purpose of protecting data integrity and security.

This article will demonstrate how to run OneHotEncoder in TEEU for processing.

## 1.1 Simulation mode

To facilitate users who do not have access to a real TEE environment, SecretFlow offers a TEEU simulation mode. This feature allows users to try out TEEU functions on an ordinary machine.
Code writing and usage in the simulation mode are almost same with the non-simulation mode, so it is recommended to use the simulation mode for quick experimental verification first.

Note that since the real TEE environment is not used, the simulation mode lacks security features that depend on the TEE environment, such as remote attestation and memory encryption isolation, and cannot protect data integrity and confidentiality. Simulation mode is not secure and should not be used in production, keep this in mind.

### Pre-work

#### Understand the SecretFlow deployment of multi-ray cluster mode

For security reasons, Ray running in TEE is an independent cluster, so currently SecretFlow only supports the use of TEEU in multiple Ray cluster mode. You can read the [SecretFlow Deployment Documentation](../../getting_started/deployment.md#production) in advance to understand the deployment of multiple Ray clusters.

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

### Example: OneHotEncoder in TEEU

Next, we will demonstrate how to combine data from multiple parties in TEEU, and then use OneHotEncoder to process it.

#### Example code

Assuming that Alice and Bob have the same feature space, but the sample space does not overlap with each other, and each has some user features, Alice and Bob intend to use TEEU to safely combine their samples and use OneHotEncoder to process data's label. At the same time, Carol acts as the provider of TEEU.

The core code of the above case is as follows.

```python
import secretflow as sf
import numpy as np

def onehot_encoder(data_tee):
    """
    Cancat the input data, then process it with OneHotEncoder.
    """
    from sklearn.preprocessing import OneHotEncoder

    data = np.concatenate((data_tee))
    enc = OneHotEncoder(sparse=False)
    result = enc.fit_transform(data.reshape(-1, 1))
    return result

def gen_data():
    """
    Generate random classified data for simulation.
    """
    from sklearn.datasets import make_classification

    num_classes = 3
    # Only used y in OneHotEncoder
    x, y = make_classification(n_samples=1000, n_informative=5,
                           n_classes=num_classes)
    y_str = y.astype(str)
    for i in range(num_classes):
        label = 'label_' + str(i)
        index = np.where(y_str==str(i))
        y_str[index] = label
    return x, y_str

alice = sf.PYU('alice')
bob = sf.PYU('bob')

# Alice generates its samples.
_, data_a = alice(gen_data, num_returns=2)()
# Bob generates its samples.
_, data_b = bob(gen_data, num_returns=2)()

from secretflow.device import TEEU

# mrenclave can be omitted in simulation mode.
teeu = TEEU('carol', mr_enclave='')

# Transfer data to teeu.
data_a_teeu = data_a.to(teeu, allow_funcs=onehot_encoder)
data_b_teeu = data_b.to(teeu, allow_funcs=onehot_encoder)

# Run onehot_encoder.
res = teeu(onehot_encoder)([data_a_teeu, data_b_teeu])
result = sf.reveal(res)
print('Label_Encoder: ', result)

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
import numpy as np

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

def onehot_encoder(data_tee):
    """
    Cancat the input data, then process it with OneHotEncoder.
    """
    from sklearn.preprocessing import OneHotEncoder

    data = np.concatenate((data_tee))
    enc = OneHotEncoder(sparse=False)
    result = enc.fit_transform(data.reshape(-1, 1))
    return result


def gen_data():
    """
    Generate random classified data for simulation.
    """
    from sklearn.datasets import make_classification

    num_classes = 3
    # Only used y in OneHotEncoder
    x, y = make_classification(n_samples=1000, n_informative=5,
                           n_classes=num_classes)
    y_str = y.astype(str)
    for i in range(num_classes):
        label = 'label_' + str(i)
        index = np.where(y_str==str(i))
        y_str[index] = label
    return x, y_str

alice = sf.PYU('alice')
bob = sf.PYU('bob')

# Alice generates its samples.
_, data_a = alice(gen_data, num_returns=2)()
# Bob generates its samples.
_, data_b = bob(gen_data, num_returns=2)()

from secretflow.device import TEEU

# mrenclave can be omitted in simulation mode.
teeu = TEEU('carol', mr_enclave='')

# Transfer data to teeu.
data_a_teeu = data_a.to(teeu, allow_funcs=onehot_encoder)
data_b_teeu = data_b.to(teeu, allow_funcs=onehot_encoder)

# Run onehot_encoder.
res = teeu(onehot_encoder)([data_a_teeu, data_b_teeu])
result = sf.reveal(res)
print('Label_Encoder: ', result)

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
import numpy as np

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

# Connect to bob's ray
sf.init(
    address='192.168.0.20:10000',
    cluster_config=cluster_config,
    party_key_pair=party_key_pair,
    auth_manager_config=auth_manager_config,
    tee_simulation=True,
)

def onehot_encoder(data_tee):
    """
    Cancat the input data, then process it with OneHotEncoder.
    """
    from sklearn.preprocessing import OneHotEncoder

    data = np.concatenate((data_tee))
    enc = OneHotEncoder(sparse=False)
    result = enc.fit_transform(data.reshape(-1, 1))
    return result


def gen_data():
    """
    Generate random classified data for simulation.
    """
    from sklearn.datasets import make_classification

    num_classes = 3
    # Only used y in OneHotEncoder
    x, y = make_classification(n_samples=1000, n_informative=5,
                           n_classes=num_classes)
    y_str = y.astype(str)
    for i in range(num_classes):
        label = 'label_' + str(i)
        index = np.where(y_str==str(i))
        y_str[index] = label
    return x, y_str

alice = sf.PYU('alice')
bob = sf.PYU('bob')

# Alice generates its samples.
_, data_a = alice(gen_data, num_returns=2)()
# Bob generates its samples.
_, data_b = bob(gen_data, num_returns=2)()

from secretflow.device import TEEU

# mrenclave can be omitted in simulation mode.
teeu = TEEU('carol', mr_enclave='')

# Transfer data to teeu.
data_a_teeu = data_a.to(teeu, allow_funcs=onehot_encoder)
data_b_teeu = data_b.to(teeu, allow_funcs=onehot_encoder)

# Run onehot_encoder.
res = teeu(onehot_encoder)([data_a_teeu, data_b_teeu])
result = sf.reveal(res)
print('Label_Encoder: ', result)

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

def onehot_encoder(data_tee):
    """
    Cancat the input data, then process it with OneHotEncoder.
    """
    from sklearn.preprocessing import OneHotEncoder

    data = np.concatenate((data_tee))
    enc = OneHotEncoder(sparse=False)
    result = enc.fit_transform(data.reshape(-1, 1))
    return result


def gen_data():
    """
    Generate random classified data for simulation.
    """
    from sklearn.datasets import make_classification

    num_classes = 3
    # Only used y in OneHotEncoder
    x, y = make_classification(n_samples=1000, n_informative=5,
                           n_classes=num_classes)
    y_str = y.astype(str)
    for i in range(num_classes):
        label = 'label_' + str(i)
        index = np.where(y_str==str(i))
        y_str[index] = label
    return x, y_str

alice = sf.PYU('alice')
bob = sf.PYU('bob')

# Alice generates its samples.
_, data_a = alice(gen_data, num_returns=2)()
# Bob generates its samples.
_, data_b = bob(gen_data, num_returns=2)()

from secretflow.device import TEEU

# mrenclave can be omitted in simulation mode.
teeu = TEEU('carol', mr_enclave='')

# Transfer data to teeu.
data_a_teeu = data_a.to(teeu, allow_funcs=onehot_encoder)
data_b_teeu = data_b.to(teeu, allow_funcs=onehot_encoder)

# Run onehot_encoder.
res = teeu(onehot_encoder)([data_a_teeu, data_b_teeu])
result = sf.reveal(res)
print('Label_Encoder: ', result)

```

Then we run the script with the following command.

```bash
cd /root/occlum_instance
openssl genrsa -3 -out private_key.pem 3072
openssl rsa -in private_key.pem -pubout -out public_key.pem
occlum build --sgx-mode sim --sign-key private_key.pem
occlum run /bin/python /root/demo.py
```

## 1.2 Non-simulation mode

When it is necessary to use the real TEE environment to protect the confidentiality and integrity of the data in the computing process, the user needs to enable the non-simulation mode, and at this time, the security mechanisms provided by the TEE such as remote attestation and memory encryption will be enabled. To enable the non-simulation mode, the user needs to have the TEE hardware supported by the current SecretFlow TEEU. Currently, SecretFlow only supports Intel SGX2.0, and more TEE types will be supported in the future.

Please check [Non-simulation](../teeu.md#summary) for running in non-simulation mode.
