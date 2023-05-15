# TEEU Getting Started Guide

Trusted Execution Environment (TEE) is a hardware-based privacy preserving technology. It ensures the authenticity of executing code, the integrity of runtime state (such as CPU registers, memory, and sensitive I/O), and the confidentiality of code, data, and runtime state stored in memory. In addition, it should be possible to provide remote attestation to third parties to prove its reliability.

TEEU (`TEE` processing `U`nit) is a TEE device in SecretFlow. Through TEEU, users can conveniently put data in TEE for calculation, and achieve the purpose of protecting data integrity and security.
As a member of the SecretFlow device layer, TEEU enriches the device layer capabilities of SecretFlow, and brings more combinations and possibilities to the mixed computing of plaintext and ciphertext. For example, the weight aggregation operation of the horizontal federation model can be placed in TEEU to protect the safety of the gradient aggregation process, or use TEEU to quickly generate the offline factors of the SPDZ protocol to accelerate MPC. More possibilities are waiting for your exploration.

At present, SecretFlow provides two modes of using TEEU, simulation and non-simulation. It is recommended that you start with the simulation mode, which will help you get started quickly and successfully.

## 1.1 Simulation mode

To facilitate users who do not have access to a real TEE environment, SecretFlow offers a TEEU simulation mode. This feature allows users to try out TEEU functions on an ordinary machine.
Code writing and usage in the simulation mode are almost same with the non-simulation mode, so it is recommended to use the simulation mode for quick experimental verification first.

Note that since the real TEE environment is not used, the simulation mode lacks security features that depend on the TEE environment, such as remote attestation and memory encryption isolation, and cannot protect data integrity and confidentiality. Simulation mode is not secure and should not be used in production, keep this in mind.

### Pre-work

#### Understand the SecretFlow deployment of multi-ray cluster mode

For security reasons, Ray running in TEE is an independent cluster, so currently SecretFlow only supports the use of TEEU in multiple Ray cluster mode. You can read the [SecretFlow Deployment Documentation](../getting_started/deployment.md#production) in advance to understand the deployment of multiple Ray clusters.

#### Prepare to run the simulated TEEU machine

At present, SecretFlow TEEU only provides docker images. Due to some limitations of the basic technology, TEE programs currently require a large amount of memory to run successfully. You need to ensure that the available memory for the docker container is at least greater than 30GB.

#### Deploy AuthManager

AuthManager is the module responsible for authorization management.

1. Download the docker image
```shell
docker pull secretflow/authmanager-ubuntu-sim-release:latest
```

2. Enter the docker image
```shell
docker run -it --net host secretflow/authmanager-ubuntu-sim-release:latest
```

3. (Optional) Configure TLS

AuthManager enables TLS by default. If you only use it for local simulation, you can turn off TLS by set `enable_tls` to `false` in `/root/occlum_release/config.yaml`.

4. Start the service

```shell
occlum run /bin/auth-manager --config_path /host/config.yaml
```
The default port is 8835. Feel free to modify the `port` in config.yaml if port conflicts.

### Example: TEEU secure aggregation

In federated learning scenarios, how to safely aggregate data from multiple parties is a common problem. Using TEEU can easily obtain the secure aggregation capability. The following example demonstrates how to use TEEU for secure aggregation.
We assume that Alice and Bob are data providers, and Carol is a provider of TEEU to provide secure aggregation computing services.

#### Write Secure Aggregation Code

The following code demonstrates that Alice and Bob each generate a numpy array, and then send it to the TEEU held by Carol for secure aggregation.
Finally, in order to verify the correctness, the results of the original value plaintext aggregation and TEEU aggregation were compared, and the results of the two should be consistent.

```python
import numpy as np

def average(data):
    return np.average(data, axis=1)

alice = sf.PYU('alice')
bob = sf.PYU('bob')

from secretflow.device import TEEU

# mrenclave can be omitted in simulation mode.
teeu = TEEU('carol', mr_enclave='')

a = alice(lambda: np.random.rand(4, 3))()
b = bob(lambda: np.random.rand(4, 3))()

# Transfer data to teeu.
a_teeu = a.to(teeu, allow_funcs=average)
b_teeu = b.to(teeu, allow_funcs=average)

# TEEU runs average.
avg_val = teeu(average)([a_teeu, b_teeu])
print(sf.reveal(avg_val))

a = sf.reveal(a)
b = sf.reveal(b)
np.testing.assert_equal(avg_val, average([a, b]) )
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

Add the SecretFlow initialization related code in front of the secure aggregation code to get the following code.
First, you need to modify the configuration items in the code.
- The code assumes that Alice's communication address is 192.168.0.10:10001, please modify it according to the actual situation
- You need to fill in the correct `auth_manager_config`
  - `host` is the listening address of the AuthManager service
  - `ca_cert` is the CA certificate address of AuthManager, if AuthManager does not start with TLS, no configuration is required.

Suppose we save the code as demo.py, and then execute `python demo.py` on Alice's machine.

```python
import secretflow as sf

cluster_config = {
    'parties': {
        'alice': {'address': '192.168.0.10:10001', 'listen_address': '0.0.0.0:10001'},
        'bob': {'address': '192.168.0.20:10001', 'listen_address': '0.0.0.0:10001'},
        'carol': {'address': '192.168.0.30:10001', 'listen_address': '0.0.0.0:10001'},
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

# Connect to Alice's ray
sf.init(
    address='192.168.0.10:10000',
    cluster_config=cluster_config,
    party_key_pair=party_key_pair,
    auth_manager_config=auth_manager_config,
    tee_simulation=True,
)

import numpy as np

def average(data):
    return np.average(data, axis=1)

from secretflow.device import TEEU

alice = sf.PYU('alice')
bob = sf.PYU('bob')
# mrenclave can be omitted in simulation mode.
teeu = TEEU('carol', mr_enclave='')

a = alice(lambda: np.random.rand(4, 3))()
b = bob(lambda: np.random.rand(4, 3))()

# Transfer data to teeu.
a_teeu = a.to(teeu, allow_funcs=average)
b_teeu = b.to(teeu, allow_funcs=average)

# TEEU runs average.
avg_val = teeu(average)([a_teeu, b_teeu])
print(sf.reveal(avg_val))


a = sf.reveal(a)
b = sf.reveal(b)
np.testing.assert_equal(avg_val, average([a, b]) )
```

#### Bob runs the code

1. Start the ray master node

You should modify the following command to match the actual situation, as it currently assumes that Bob's Ray master node is listening at 192.168.0.20:10000.
```bash
ray start --head --node-ip-address="192.168.0.20" --port="10000" --include-dashboard=False --disable-usage-stats
```

2. Generate a public-private key pair

As Bob's data needs to be encrypted and sent to TEEU, it is imperative to generate a pair of public and private keys. Below, you may find the code that, upon execution, generates the public and private keys, which will be stored in the current directory in PEM format as "private_key.pem" and "public_key.pem", respectively.
```bash
openssl genrsa -3 -out private_key.pem 3072
openssl rsa -in private_key.pem -pubout -out public_key.pem
```

3. Run the code

Similar to Alice, add the SecretFlow initialization code in front of the secure aggregation code to get the following code.
First, you need to modify the configuration items in the code.

- The code assumes that Alice's communication address is 192.168.0.20:10001, please modify it according to the actual situation
- You need to fill in the correct `auth_manager_config`
- `host` is the listening address of the AuthManager service
- `ca_cert` is the CA certificate address of AuthManager, if AuthManager does not start tls, no configuration is required.

Suppose we save the code as demo.py, and then execute `python demo.py` on Bob's machine.

```python
import secretflow as sf

cluster_config = {
    'parties': {
        'alice': {'address': '192.168.0.10:10001', 'listen_address': '0.0.0.0:10001'},
        'bob': {'address': '192.168.0.20:10001', 'listen_address': '0.0.0.0:10001'},
        'carol': {'address': '192.168.0.30:10001', 'listen_address': '0.0.0.0:10001'},
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

def average(data):
    return np.average(data, axis=1)

from secretflow.device import TEEU

alice = sf.PYU('alice')
bob = sf.PYU('bob')
# mrenclave can be omitted in simulation mode.
teeu = TEEU('carol', mr_enclave='')

a = alice(lambda: np.random.rand(4, 3))()
b = bob(lambda: np.random.rand(4, 3))()

# Transfer data to teeu.
a_teeu = a.to(teeu, allow_funcs=average)
b_teeu = b.to(teeu, allow_funcs=average)

# TEEU runs average.
avg_val = teeu(average)([a_teeu, b_teeu])
avg_val = sf.reveal(avg_val).data
print(avg_val)


a = sf.reveal(a)
b = sf.reveal(b)
np.testing.assert_equal(avg_val, average([a, b]) )
```

#### Carol runs code (executed in TEE)

Run the container firstly.

```bash
docker run -it --network host secretflow/secretflow-teeu:latest
```

---
Hint:

The display of error messages such as "Failed to open Intel SGX device", "Error, call sgx_create_enclave QE fail", "Failed to load QE3" is expected while running in simulation mode.

---

Similarly, add the SecretFlow initialization code in front of the secure aggregation code to get the following code. Unlike the previous one, Carol's code needs to run in tee, so some extra steps are required.
First, you need to modify the configuration items in the code.

1. In the code, it is assumed that Carol's communication address is 192.168.0.30:10001, please modify it according to the actual situation
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
        'alice': {'address': '192.168.0.10:10001', 'listen_address': '0.0.0.0:10001'},
        'bob': {'address': '192.168.0.20:10001', 'listen_address': '0.0.0.0:10001'},
        'carol': {'address': '192.168.0.30:10001', 'listen_address': '0.0.0.0:10001'},
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

def average(data):
    return np.average(data, axis=1)

from secretflow.device import TEEU

alice = sf.PYU('alice')
bob = sf.PYU('bob')
# mrenclave can be omitted in simulation mode.
teeu = TEEU('carol', mr_enclave='')

a = alice(lambda: np.random.rand(4, 3))()
b = bob(lambda: np.random.rand(4, 3))()

# Transfer data to teeu.
a_teeu = a.to(teeu, allow_funcs=average)
b_teeu = b.to(teeu, allow_funcs=average)

# TEEU runs average.
avg_val = teeu(average)([a_teeu, b_teeu])
print(sf.reveal(avg_val))


a = sf.reveal(a)
b = sf.reveal(b)
np.testing.assert_equal(avg_val, average([a, b]) )
```

Then we run the script with the following command.

```bash
cd /root/occlum_instance
openssl genrsa -3 -out private_key.pem 3072
openssl rsa -in private_key.pem -pubout -out public_key.pem
occlum build --sgx-mode sim --sign-key private_key.pem
occlum run /bin/python /root/demo.py
```

---
Hint:

While running on Occlum, it is anticipated to observe warning logs, such as "Fail to open /proc/self/io", "Fail to open /proc/self/statm", and "Fail to open /proc/loadavg", due to the absence of certain kernel functionalities. However, these warnings do not interfere with the proper functioning of the program.

---

## 1.2 Non-simulation mode

### About non-simulation mode

When it is necessary to use the real TEE environment to protect the confidentiality and integrity of the data in the computing process, the user needs to enable the non-simulation mode, and at this time, the security mechanisms provided by the TEE such as remote attestation and memory encryption will be enabled. To enable the non-simulation mode, the user needs to have the TEE hardware supported by the current SecretFlow TEEU. Currently, SecretFlow only supports Intel SGX2.0, and more TEE types will be supported in the future.
Currently SecretFlow TEEU only supports Occlum running on Intel SGX2.0 (more information on Occlum can be found at [https://github.com/occlum/occlum](https://github.com/occlum/occlum)), Remote attestation uses Intel SGX DCAP.

In non-simulation mode, for Carol (TEEU provider), some additional steps are required to ensure that TEEU runs on a real TEE machine, and Alice and Bob also need to perform some additional configurations.

### Prerequisites

You need to prepare a machine that supports SGX 2.0. We assume that you have installed the operating system and SGX driver on this machine, and installed the docker software.

Due to some limitations of the basic technology, the current TEE program requires a large amount of memory to run successfully. You need to ensure that the docker container can use at least 30GB of memory.

### Pre-deployment

#### **Deploy Intel SGX PCCS on SGX machine**

Refer to [Intel SGX PCCS Install Document](https://github.com/intel/SGXDataCenterAttestationPrimitives/tree/master/QuoteGeneration/pccs). You can follow the documentation to install it.

Assuming that you start the PCCS service on port 8081 (the default listening port of PPCS), subsequent documents will be described according to this port. If you modify the default listening port, remember to replace it with the actual port.

#### Deploy AuthManager
Unlike simulation mode, non-simulation mode requires the simulation flag to be turned off.

1. Download the docker image
```shell
docker pull secretflow/authmanager-ubuntu-release:latest
```

2. Enter the docker image
```shell
docker run -it --net host -v /dev/sgx_enclave:/dev/sgx/enclave -v /dev/sgx_provision:/dev/sgx/provision --privileged=true secreflow/authmanager-release-ubuntu:latest
```

3. Modify pccs configuration

Config `PCCS_URL` in `/etc/sgx_default_qcnl.conf` with the actual deployment PCCS service address and set `USE_SECURE_CERT=FALSE`.

```properties
# PCCS server address
PCCS_URL=https://localhost:8081/sgx/certification/v3/


# To accept insecure HTTPS certificate, set this option to FALSE
USE_SECURE_CERT=FALSE


# You can use Intel PCS to get quote verification collateral
#COLLATERAL_SERVICE=https://api.trustedservices.intel.com/sgx/certification/v3/


# If you use PCCS service to get quote verification collateral, you can specify which API version is to be used
# The legacy 3.0 API will return CRLs in HEX encoded DER format, while the new 3.1 API will return raw DER format
#PCCS_API_VERSION=3.1


# Maximum retry times for QCNL. If RETRY is not defined or set to 0, no retry will be performed.
# It will first wait one second and then for all forthcoming retries it will double the waiting time
# By using RETRY_DELAY you disable this exponential backoff algorithm
#RETRY_TIMES=6


# Sleep this amount of seconds before each retry when a transfer has failed with a transient error
#RETRY_DELAY=10
```

Modify `ua_dcap_pccs_url` configuration in `/root/occlum_release/image/etc/kubetee/unified_attestation.json` to the actual PCCS service address

```json
{
    "ua_ias_url": "",
    "ua_ias_spid": "",
    "ua_ias_apk_key": "",
    "ua_dcap_lib_path": "",
    "ua_dcap_pccs_url": "https://localhost:8081/sgx/certification/v3/",
    "ua_uas_url": "",
    "ua_uas_app_key": "",
    "ua_uas_app_secret": ""
}
```

4. Configure TLS

We recommend that you enable TLS. For information on how to configure this feature, please refer to [AuthManager](https://github.com/SecretFlow/authmanager).

5. Generate a pair of public and private keys, and then use the following command to build

You first need to generate a pair of public and private keys, and then use the following command to build the release version of occlum.
You can refer to the following scripts to generate public and private keys. The generated public and private keys are stored in private_key.pem and public_key.pem in the current directory. Please keep your private key safe and do not disclose it to others.

```bash
openssl genrsa -3 -out private_key.pem 3072
openssl rsa -in private_key.pem -pubout -out public_key.pem
```

After generating the public and private keys, use the private key to build.
```json
occlum build -f --sign-key private_key.pem
```

6. Run the service
```shell
occlum run /bin/auth-manager --config_path /host/config.yaml
```

The default port is 8835. If the port conflicts, you can add the `--port` parameter to specify the port number.

```shell
occlum run /bin/auth-manager --config_path /host/config.yaml
```

6. Get the mrenclave of AuthManager

Execute the following command to obtain the mrenclave of AuthManager. mrenclave can be understood as a metric representing AuthManager code, data, and operating environment. The output is a string of hexadecimal strings, you can save it and use it in the next step.

```bash
occlum print mrenclave
```

### Example - TEEU Secure Aggregation
In federated learning scenarios, how to safely aggregate data from multiple parties is a common problem. Using TEEU can easily obtain the secure aggregation capability. The following example demonstrates how to use TEEU for secure aggregation.
We assume that Alice and Bob are data providers, and Carol is a provider of TEEU to provide secure aggregation computing services.

#### Write Secure Aggregation Code
The following code demonstrates that Alice and Bob each generate a numpy array, and then send it to the TEEU held by Carol for safe aggregation (averaging).
Finally, in order to verify the correctness, the results of the original value plaintext aggregation and TEEU aggregation were compared, and the results of the two should be consistent.

```python
import numpy as np

def average(data):
    return np.average(data, axis=1)

from secretflow.device import TEEU

alice = sf.PYU('alice')
bob = sf.PYU('bob')
teeu = TEEU('carol', mr_enclave='mrenclave_of_teeu')

a = alice(lambda: np.random.rand(4, 3))()
b = bob(lambda: np.random.rand(4, 3))()

# Transfer data to teeu.
a_teeu = a.to(teeu, allow_funcs=average)
b_teeu = b.to(teeu, allow_funcs=average)

# TEEU runs average.
avg_val = teeu(average)([a_teeu, b_teeu])
print(sf.reveal(avg_val))

a = sf.reveal(a)
b = sf.reveal(b)
np.testing.assert_equal(avg_val, average([a, b]) )
```

#### Carol runs code (executed in TEE)

1. Start the container

Start the SecretFlow TEE container.
```bash
docker run -it --network host --privileged -v /dev/sgx_enclave:/dev/sgx/enclave -v /dev/sgx_provision:/dev/sgx/provision secretflow/secretflow-teeu:latest
```
After entering the container, the default location is /root/occlum_instance.

2. Modify the configuration file
The configuration file is located at `/root/occlum_instance/image/etc/kubetee/unified_attestation.json` and is required by [jinzhao-attest](https://github.com/jinzhao-dev/jinzhao-attest). This file encapsulates the remote attestation process of various TEEs. The template is as follows, and replace the `ua_dcap_pccs_url` with the deployed PCCS address.

```json
{
    "ua_ias_url": "",
    "ua_ias_spid": "",
    "ua_ias_apk_key": "",
    "ua_dcap_lib_path": "",
    "ua_dcap_pccs_url": "https://localhost:8081/sgx/certification/v3/",
    "ua_uas_url": "",
    "ua_uas_app_key": "",
    "ua_uas_app_secret": "",
    "ua_policy_str_tee_platform": "",
    "ua_policy_hex_platform_hw_version": "",
    "ua_policy_hex_platform_sw_version": "",
    "ua_policy_hex_secure_flags": "",
    "ua_policy_hex_platform_measurement": "",
    "ua_policy_hex_boot_measurement": "",
    "ua_policy_str_tee_identity": "",
    "ua_policy_hex_ta_measurement": "",
    "ua_policy_hex_ta_dyn_measurement": "",
    "ua_policy_hex_signer": "",
    "ua_policy_hex_prod_id": "",
    "ua_policy_str_min_isvsvn": "",
    "ua_policy_hex_user_data": "",
    "ua_policy_bool_debug_disabled": "",
    "ua_policy_hex_hash_or_pem_pubkey": "",
    "ua_policy_hex_nonce": "",
    "ua_policy_hex_spid": ""
}
```

After that, modify `/etc/sgx_default_qcnl.conf`, replace `PCCS_URL` with the previously deployed PCCS address, and set `USE_SECURE_CERT=FALSE`.

```json
# PCCS server address
PCCS_URL=https://localhost:8081/sgx/certification/v3/

# To accept insecure HTTPS certificate, set this option to FALSE
USE_SECURE_CERT=FALSE

# You can use Intel PCS to get quote verification collateral
#COLLATERAL_SERVICE=https://api.trustedservices.intel.com/sgx/certification/v3/

# If you use PCCS service to get quote verification collateral, you can specify which API version is to be used
# The legacy 3.0 API will return CRLs in HEX encoded DER format, while the new 3.1 API will return raw DER format
#PCCS_API_VERSION=3.1

# Maximum retry times for QCNL. If RETRY is not defined or set to 0, no retry will be performed.
# It will first wait one second and then for all forthcoming retries it will double the waiting time
# By using RETRY_DELAY you disable this exponential backoff algorithm
#RETRY_TIMES=6

# Sleep this amount of seconds before each retry when a transfer has failed with a transient error
#RETRY_DELAY=10
```

3. Write the test code

Add the SecretFlow initialization related code in front of the secure aggregation code to get the following code. Carol's code needs to run in tee, so some extra steps are required.

First, you need to modify the configuration items in the code.
- In the code, it is assumed that the communication address of Carol is 192.168.0.30:10001, please modify it according to the actual situation
- You need to fill in the correct `auth_manager_config`
  - `host` is the listening address of the AuthManager service
  - `ca_cert` is the CA certificate address of AuthManager, if AuthManager does not start tls, no configuration is required. (It is recommended to enable tls)
  - `mrenclave` is the mrenclave of the AuthManager module, you should have obtained this value in the step of deploying AuthManager.

Please save the modified file as /root/occlum_instance/image/root/demo.py.

```python
# Generate tls cert and key at first.
from tls_cert import generate_self_signed_tls_certs

generate_self_signed_tls_certs()


import secretflow as sf

cluster_config = {
    'parties': {
        'alice': {
            'address': '192.168.0.10:10001',
            'listen_address': '0.0.0.0:10001'
    },
        'bob': {
            'address': '192.168.0.20:10001',
            'listen_address': '0.0.0.0:10001'
        },
        'carol': {
            'address': '192.168.0.30:10001',
            'listen_address': '0.0.0.0:10001'
        },
    },
    'self_party': 'carol'
}


auth_manager_config = {
    'host': 'host of AuthManager',
    'ca_cert': 'path_of_ca_certificate_of_AuthManager',
    'mr_enclave': 'mrenclave of AuthManager',
}

# Carol starts a local ray inside tee.
sf.init(
    address='local', 
    cluster_config=cluster_config, 
    auth_manager_config=auth_manager_config,
    _temp_dir="/host/tmp/ray",
    _plasma_directory="/tmp",
)

import numpy as np

def average(data):
    return np.average(data, axis=1)

from secretflow.device import TEEU

alice = sf.PYU('alice')
bob = sf.PYU('bob')
# Carol can omit the mrenclave.
teeu = TEEU('carol', mr_enclave='')

a = alice(lambda: np.random.rand(4, 3))()
b = bob(lambda: np.random.rand(4, 3))()

# Transfer data to teeu.
a_teeu = a.to(teeu, allow_funcs=average)
b_teeu = b.to(teeu, allow_funcs=average)

# TEEU runs average.
avg_val = teeu(average)([a_teeu, b_teeu])
print(sf.reveal(avg_val))

a = sf.reveal(a)
b = sf.reveal(b)
np.testing.assert_equal(avg_val, average([a, b]) )
```

4. build occlum

You first need to generate a pair of public and private keys, and then use the following command to build the release version of occlum.
You can refer to the following scripts to generate public and private keys. The generated public and private keys are stored in private_key.pem and public_key.pem in the current directory. Please keep your private key safe and do not disclose it to others.

```bash
openssl genrsa -3 -out private_key.pem 3072
openssl rsa -in private_key.pem -pubout -out public_key.pem
```

After generating the public and private keys, use the private key to build.

```json
occlum build -f --sign-key private_key.pem
```

5. Obtain the MRENCLAVE of TEEU

Execute the following command to get the MRENCLAVE of TEEU, which is used to characterize the metrics of TEEU code, data, and operating environment. The output is a string of hexadecimal strings, you can save it and use it in the next step.

```bash
occlum print mrenclave
```

6. Run the code

Execute the following command to run the script.
```bash
occlum run /bin/python /root/demo.py
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

Add the SecretFlow initialization related code in front of the secure aggregation code to get the following code. You need to configure the code:

- The code assumes that Alice's communication address is 192.168.0.10:10001, please modify it according to the actual situation
- You need to fill in the correct `auth_manager_config`
  - `host` is the listening address of the AuthManager service
  - `ca_cert` is the CA certificate address of AuthManager, if AuthManager does not start tls, no configuration is required. (It is recommended to enable tls)
  - `mrenclave` is the MRENCLAVE of the AuthManager module, you should have obtained this value in the step of deploying AuthManager.
- Use the MRENCLAVE of TEEU obtained earlier, fill in the correct value: `teeu = TEEU('carol', mr_enclave='mr_enclave of TEEU')`

After the configuration is complete, suppose we save the code as demo.py, and then execute `python demo.py` on Alice's machine.

```python
import secretflow as sf

cluster_config = {
    'parties': {
        'alice': {'address': '192.168.0.10:10001', 'listen_address': '0.0.0.0:10001'},
        'bob': {'address': '192.168.0.20:10001', 'listen_address': '0.0.0.0:10001'},
        'carol': {'address': '192.168.0.30:10001', 'listen_address': '0.0.0.0:10001'},
    },
    'self_party': 'alice',
}

party_key_pair = {
    'alice': {'private_key': './private_key.pem', 'public_key': './public_key.pem'}
}


auth_manager_config = {
    'host': 'host of AuthManager',
    'ca_cert': 'path_of_ca_certificate_of_AuthManager',
    'mr_enclave': 'mrenclave of AuthManager',
}

# Connect to Alice's ray
sf.init(
    address='192.168.0.10:10000',
    cluster_config=cluster_config,
    party_key_pair=party_key_pair,
    auth_manager_config=auth_manager_config,
    tee_simulation=False,
)


import numpy as np

def average(data):
    return np.average(data, axis=1)

from secretflow.device import TEEU

alice = sf.PYU('alice')
bob = sf.PYU('bob')
teeu = TEEU('carol', mr_enclave='mrenclave_of_TEEU')

a = alice(lambda: np.random.rand(4, 3))()
b = bob(lambda: np.random.rand(4, 3))()
a_teeu = a.to(teeu, allow_funcs=average)
b_teeu = b.to(teeu, allow_funcs=average)

avg_val = teeu(average)([a_teeu, b_teeu])
print(sf.reveal(avg_val))


a = sf.reveal(a)
b = sf.reveal(b)
np.testing.assert_equal(avg_val, average([a, b]) )
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

Similar to Alice, add the SecretFlow initialization code in front of the secure aggregation code to get the following code. You need to configure the code:
- In the code, it is assumed that Bob’s communication address is 192.168.0.20:10001, please modify it according to the actual situation
- You need to fill in the correct `auth_manager_config`
  - `host` is the listening address of the AuthManager service
  - `ca_cert` is the CA certificate address of AuthManager, if AuthManager does not start tls, no configuration is required. (It is recommended to enable tls)
  - `mrenclave` is the MRENCLAVE of the AuthManager module, you should have obtained this value in the step of deploying AuthManager.
- Use the MRENCLAVE of TEEU obtained earlier, fill in the correct value: `teeu = TEEU('carol', mr_enclave='mr_enclave of TEEU')`

After the configuration is complete, suppose we save the code as demo.py, and then execute `python demo.py` on Bob's machine.

```python
import secretflow as sf

cluster_config = {
    'parties': {
        'alice': {'address': '192.168.0.10:10001', 'listen_address': '0.0.0.0:10001'},
        'bob': {'address': '192.168.0.20:10001', 'listen_address': '0.0.0.0:10001'},
        'carol': {'address': '192.168.0.30:10001', 'listen_address': '0.0.0.0:10001'},
    },
    'self_party': 'bob',
}

party_key_pair = {
    'bob': {'private_key': './private_key.pem', 'public_key': './public_key.pem'}
}


auth_manager_config = {
    'host': 'host of AuthManager',
    'ca_cert': 'path_of_ca_certificate_of_AuthManager',
    'mr_enclave': 'mrenclave of AuthManager',
}

# Connect to Bob's ray
sf.init(
    address='192.168.0.20:10000',
    cluster_config=cluster_config,
    party_key_pair=party_key_pair,
    auth_manager_config=auth_manager_config,
    tee_simulation=False,
)


import numpy as np

def average(data):
    return np.average(data, axis=1)


alice = sf.PYU('alice')
bob = sf.PYU('bob')
teeu = TEEU('carol', mr_enclave='mrenclave_of_TEEU')

a = alice(lambda: np.random.rand(4, 3))()
b = bob(lambda: np.random.rand(4, 3))()
a_teeu = a.to(teeu, allow_funcs=average)
b_teeu = b.to(teeu, allow_funcs=average)

avg_val = teeu(average)([a_teeu, b_teeu])
print(sf.reveal(avg_val))


a = sf.reveal(a)
b = sf.reveal(b)
np.testing.assert_equal(avg_val, average([a, b]) )
```

### Summary
This chapter demonstrates how to use TEEU on a real SGX 2.0 machine. The main difference between non-simulation mode and simulation mode is as follows.

1. Configure remote attestation.
  - Deploy DCAS
  - Configure `/etc/sgx_default_qcnl.conf`
  - Configure `image/etc/kubetee/unified_attestation.json`
2. To start the AuthManager and SecretFlow TEE images, it is necessary to mount the SGX-related devices. E.g.,
```bash
docker run -it --network host --privileged -v /dev/sgx_enclave:/dev/sgx/enclave -v /dev/sgx_provision:/dev/sgx/provision secretflow/secretflow-teeu:latest
```
3. Obtain and fill in the measurement values (MRENCLAVE) of AuthManager and TEEU.
  - Fill in the mrenclave of AuthManager in `auth_manager_config` of `sf.init()`.
  - Fill in the MRENCLAVE of TEEU when constructing its instance by `teeu = sf.TEEU(..., mr_enclave='MRENCLAVE_OF_TEEU')` .
4. Set `tee_simulation=False` in `sf.init()`.
5. Build occlum without arg `--sgx-mode sim`.

### More examples

[XGBoost in TEEU](./teeu_xgboost.md)

## 1.3 Advanced topics

If you want to know more about TEEU, please read the developer-oriented document [TEEU](../developer/design/teeu.md), which introduces the principles and design ideas behind TEEU.
