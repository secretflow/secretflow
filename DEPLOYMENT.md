# Deployment

SecretFlow can be deployed on a single host or on multiple nodes.

## Standalone Mode
Use `secretflow.init` directly to run secretflow in standalone mode.

```python
>>> import secretflow as sf
>>> sf.init(['alice', 'bob', 'carol'], num_cpus=8, log_to_driver=True)
```
## Cluster Mode
The following is an example showing how to build a cluster consisting of alice and bob on multiple nodes.

### Start head node
Start a head node on your first machine with the tag "alice".

---
**NOTE**

1. Remember to use the real ip and port instead.

2. You can refer to [Ray TLS](https://docs.ray.io/en/latest/ray-core/configure.html#tls-authentication) for servercert.pem, serverkey.pem and cacert.pem.

3. The following section `Suggestions for production` explains `RAY_SECURITY_CONFIG_PATH` and config.yml.

4. It's ok to remove these environments for testing if in an intranet.

5. `{"alice": 8}` means that alice can run up to 8 workers at the same time. Just feel free to change it if you like.
---

```bash
RAY_DISABLE_REMOTE_CODE=true \
RAY_SECURITY_CONFIG_PATH=config.yml \
RAY_USE_TLS=1 \
RAY_TLS_SERVER_CERT=servercert.pem \
RAY_TLS_SERVER_KEY=serverkey.pem \
RAY_TLS_CA_CERT=cacert.pem \
ray start --head --node-ip-address="ip" --port="port" --resources='{"alice": 8}' --include-dashboard=False --disable-usage-stats
```

Head node starts successfully if you see "Ray runtime started." in the screen output.

Now we have a cluster with a head node only, let us start more nodes.

### Start other nodes
Start a node with the tag "bob" on another machine. The node will connect to the head node and join the cluster.

---
**Note**

Replace `ip:port` with the `node-ip-address` and `port` of head node please.

---

```bash
RAY_DISABLE_REMOTE_CODE=true \
RAY_SECURITY_CONFIG_PATH=config.yml \
RAY_USE_TLS=1 \
RAY_TLS_SERVER_CERT=servercert.pem \
RAY_TLS_SERVER_KEY=serverkey.pem \
RAY_TLS_CA_CERT=cacert.pem \
ray start --address="ip:port" --resources='{"bob": 8}' --disable-usage-stats
```

The node starts successfully if you see "Ray runtime started." in the screen output. 

You can repeat the step above to start more nodes with using other parties as resources tag.

### Start SecretFlow
Now you can start SecretFlow and run your code.

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

### (optional) How to shut down the cluster
In some cases you would like to shut down the cluster, the following command will help you.
Remember to run the command on all machines.

Note that all ray processors on the machine will be stopped, which means all ray
clusters will be stopped.

```bash
ray stop
```

### (optional) How to setup a SPU in cluster mode

`SPU` consists of multi workers on different nodes.
For performance reasons, the major part of SPU is written in C++.
SPU is based on Brpc, which indicates it has a separated service mesh independent of Ray's networking.
In a word, you need to assign different ports for the SPU for now.
We are working on merging them.

A typical SPU config:
```python
import spu
import secteflow as sf

cluster_def={
    'nodes': [
        {
            'party': 'alice',
            'id': '0',
            # Use the address and port of alice instead.
            # Please choose a unused port.
            'address': 'address:port',
        },
        {
            'party': 'bob',
            'id': '1',
            # Use the ip and port of bob instead.
            # Please choose a unused port.
            'address': 'address:port',
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

For more configurations of SPU, please refer to [SPU config](https://spu.readthedocs.io/en/beta/reference/runtime_config.html)

---
**Note**

You will see the usage of setup a spu in many toturials. But
be careful that it works only in standalone mode because `sf.utils.testing.cluster_def` use `127.0.0.1` as the default ip.

```python
>>> spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))
```

---



### Suggestions for production
SecretFlow use `ray` as its distribution system. 
You may need to do some more configuration for higher security when using it in production.
The following actions can help improve security features.

1. Enable tls Authentication.

    Ray can be configured to use TLS on its gRPC channels, for more, please refer to [Ray TLS](https://docs.ray.io/en/latest/ray-core/configure.html#tls-authentication).

2. Forbidden on-fly remote.

    `Remote` is one of the most important features of ray, but it may become dangerous when unexpected functions are injected into your node without knowing. You can set environment `RAY_DISABLE_REMOTE_CODE=true` to close the remote execution.

3. Enhanced serialization/deserialization.

    Ray uses `pickle` in serialization/deserialization which is vulnerable. You can set environment `RAY_SECURITY_CONFIG_PATH=config.yml` to specify an allowlist to restrict serializable objects.
    An example of config.yml could be
    ```yaml
    pickle_whitelist:
        builtins:
        - type
        numpy:
        - dtype
        numpy.core.numeric:
        - '*'
    ```
    You should not use this demo YAML directly. Configure it to your actual needs.
