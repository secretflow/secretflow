# Deployment

## Simulation
SecretFlow is designed for fast simulation on a single host or on multiple nodes with single ray cluster.

**Note**

SecretFlow with single ray cluster is for simulation only. Please refer to `production` section below for production.

---
### Standalone mode for simulation
Use `secretflow.init` directly to run secretflow in standalone mode.

```python
>>> import secretflow as sf
>>> sf.init(['alice', 'bob', 'carol'], address='local')
```
### Cluster mode for simulation
The following is an example showing how to build a cluster consisting of alice and bob on multiple nodes.

#### Start head node
Start a head node on your first machine with the tag "alice".

---
**NOTE**

1. Remember to use the real ip and port instead.

2. `{"alice": 16}` means that alice can run up to 16 workers at the same time. Just feel free to change it if you like.
---

```bash
ray start --head --node-ip-address="ip" --port="port" --resources='{"alice": 16}' --include-dashboard=False --disable-usage-stats
```

Head node starts successfully if you see "Ray runtime started." in the screen output.

Now we have a cluster with a head node only, let us start more nodes.

#### Start other nodes
Start a node with the tag "bob" on another machine. The node will connect to the head node and join the cluster.

---
**Note**

Replace `ip:port` with the `node-ip-address` and `port` of head node please.

---

```bash
ray start --address="ip:port" --resources='{"bob": 16}' --include-dashboard=False --disable-usage-stats
```

The node starts successfully if you see "Ray runtime started." in the screen output. 

You can repeat the step above to start more nodes with using other parties as resources tag as you like.

#### Start SecretFlow
Now you can start SecretFlow and run your code.
Fill `address` of `sf.init` with the `node-ip-address` and `port` of head node please.

```python
>>> import secretflow as sf
# Replace with the `node-ip-address` and `port` of head node.
>>> sf.init(address='ip:port')
>>> alice = sf.PYU('alice')
>>> bob = sf.PYU('bob')
>>> alice(lambda x : x)(2)
<secretflow.device.device.pyu.PYUObject object at 0x7fe932a1a640>
>>> bob(lambda x : x)(2)
<secretflow.device.device.pyu.PYUObject object at 0x7fe6fef03250>
```

#### (optional) How to shut down the cluster
In some cases you would like to shut down the cluster, the following command will help you.
Remember to run the command on all machines.

Note that all ray processors on the machine will be stopped, which means all ray
clusters will be stopped.

```bash
ray stop
```

#### (optional) How to setup a SPU in cluster mode

`SPU` consists of multi workers on different nodes.
For performance reasons, the major part of SPU is written in C++.
SPU is based on Brpc, which indicates it has a separated service mesh independent of Ray's networking.
In a word, you need to assign different ports for the SPU for now.
We are working on merging them.

A typical SPU config is as follows.

---
**Tips**

1. Replace `ip:port` in `sf.init` with the `node-ip-address` and `port` of head node please.
2. Fill `address` of `alice` with the ip which can be accessed by `bob` and choose **an unused port**. 
3. Fill `address` of `bob` with the ip which can be accessed by `alice` and choose **an unused port**. 

---
```python
import spu
import secretflow as sf

# Use ray head adress please.
sf.init(parties=['alice', 'bob'], address='ip:port')

cluster_def={
    'nodes': [
        {
            'party': 'alice',
            'id': '0',
            # Use the ip and port of alice instead.
            # Please choose an unused port.
            'address': 'address:port',
            'listen_addr': '0.0.0.0:port'
        },
        {
            'party': 'bob',
            'id': '1',
            # Use the ip and port of bob instead.
            # Please choose an unused port.
            'address': 'address:port',
            'listen_addr': '0.0.0.0:port'
        },
    ],
    'runtime_config': {
        'protocol': spu.spu_pb2.SEMI2K,
        'field': spu.spu_pb2.FM128,
        'sigmoid_mode': spu.spu_pb2.RuntimeConfig.SIGMOID_REAL,
    }
}

spu = sf.SPU(cluster_def=cluster_def)
```

For more configurations of SPU, please refer to [SPU config](https://www.secretflow.org.cn/docs/spu/en/reference/runtime_config.html)

---
**Note**

You will see the usage of setup a spu in many toturials. But
be careful that it works only in standalone mode because `sf.utils.testing.cluster_def` use `127.0.0.1` as the default ip.

```python
>>> spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob', 'carol']))
```
---

## Production

SecretFlow provides multi controller mode for production with enhanced security.
The following will guide you to deploy SecretFlow for production. 

### Setup a SecretFlow cluster crossing silo

A SecretFlow cluster for production consists of serveral ray clusters, and every party has its own ray cluster.

The following is an example showing how to build a cluster consisting of alice and bob for production.
#### Start SecretFlow on the node of `alice`

`alice` starts its ray cluster firstly.
```bash
ray start --head --node-ip-address="ip" --port="port" --include-dashboard=False --disable-usage-stats
```
Head node starts successfully if you see "Ray runtime started." in the screen output. 

Then `alice` initializes SecretFlow with a cluster config.

---
**Tips**
1. Replace `ip:port` in `sf.init` with the `node-ip-address` and `port` of head node please.
2. Fill `address` of `alice` with the address which can be accessed by `bob`. Remember to choose an unused port. 
3. Fill `address` of `bob` with the address which can be accessed by `alice`. Remember to choose an unused port. 
4. Note that `self_party` is alice.
---
```python
cluster_config ={
    'parties': {
        'alice': {
            # replace with alice's real address.
            'address': 'ip:port',
            'listen_addr': '0.0.0.0:port'
        },
        'bob': {
            # replace with bob's real address.
            'address': 'ip:port',
            'listen_addr': '0.0.0.0:port'
        },
    },
    'self_party': 'alice'
}

sf.init(address='ip:port', cluster_config=cluster_config)

# your code to run.
```

#### Start SecretFlow on the node of `bob`

`bob` starts its ray cluster firstly.
```bash
ray start --head --node-ip-address="ip" --port="port" --include-dashboard=False --disable-usage-stats
```
Head node starts successfully if you see "Ray runtime started." in the screen output. 


Then `bob` initializes SecretFlow with a cluster config almost same as `alice` except for `self_party`.

---
**Tips**
1. Replace `ip:port` in `sf.init` with the `node-ip-address` and `port` of head node please.
2. Fill `address` of `alice` with the address which can be accessed by `bob`. Remember to choose an unused port. 
3. Fill `address` of `bob` with the address which can be accessed by `alice`. Remember to choose an unused port. 
4. Note that `self_party` is `bob`. 
---

```python

cluster_config ={
    'parties': {
        'alice': {
            # replace with alice's real address.
            'address': 'ip:port',
            'listen_addr': '0.0.0.0:port'
        },
        'bob': {
            # replace with bob's real address.
            'address': 'ip:port',
            'listen_addr': '0.0.0.0:port'
        },
    },
    'self_party': 'bob'
}

sf.init(address='ip:port', cluster_config=cluster_config)

# your code to run.
```

### How to setup SPU for production

Just same as simulation, please refer to the previous for details.

### Suggestions for production

1. Enable tls Authentication.

    SecretFlow can be configured to use TLS on cross-silo gRPC channels.

    An example for alice.
    ```python
    tls_config = {
        "cert": {
            # Alice's cert and key.
            "ca_cert": "cacert.pem",
            "cert": "servercert.pem",
            "key": "serverkey.pem",
        },
        "client_certs": {
            # peer's cert.
            "bob":  {
                "ca_cert": "bob's cacert.pem",
                "cert": "bob's servercert.pem",
            }
        }
    }

    sf.init(address='ip:port', 
            cluster_config=cluster_config, 
            tls_config=tls_config
    )
    ```

    An example for bob.
    ```python
    tls_config = {
        "cert": {
            # Bob's cert and key.
            "ca_cert": "cacert.pem",
            "cert": "servercert.pem",
            "key": "serverkey.pem",
        },
        "client_certs": {
            # peer's cert.
            "alice":  {
                "ca_cert": "alice's cacert.pem",
                "cert": "alice's servercert.pem",
            }
        }
    }

    sf.init(address='ip:port', 
            cluster_config=cluster_config, 
            tls_config=tls_config
    )
    ```

2. Enhanced serialization/deserialization.

    SecretFlow uses `pickle` in serialization/deserialization which is vulnerable. You can set `cross_silo_serializing_allowed_list` when init  SecretFlow to specify an allowlist to restrict serializable objects.
    An example could be （**You should not use this demo directly. Configure it to your actual needs.**）
    ```python
    allowed_list =  {
        "numpy.core.numeric": ["*"],
        "numpy": ["dtype"],
    }

    sf.init(address='ip:port', 
            cluster_config=cluster_config, 
            cross_silo_serializing_allowed_list=allowed_list
    )
    ```
