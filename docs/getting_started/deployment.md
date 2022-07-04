# Deployment

Secretflow can be deployed on a single host or on multiple nodes.

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

> Remember to use the real ip and port instead.

> You can refert to [Ray TLS](https://docs.ray.io/en/latest/ray-core/configure.html#tls-authentication) for servercert.pem, serverkey.pem and cacert.pem.

> The following section `Suggestions for production` explains `RAY_SECURITY_CONFIG_PATH` and config.yml.

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

You will see redis_password in the screen output, save it to use in the next step.

### Start other nodes
Start a node with the tag "bob" on another machine. 

Replace `ip:port` with your real ip and port please.
Replace `redis_password` with the correct value saved previous.

```bash
RAY_DISABLE_REMOTE_CODE=true \
RAY_SECURITY_CONFIG_PATH=config.yml \
RAY_USE_TLS=1 \
RAY_TLS_SERVER_CERT=servercert.pem \
RAY_TLS_SERVER_KEY=serverkey.pem \
RAY_TLS_CA_CERT=cacert.pem \
ray start --address="ip:port" --redis-password='redis_password' --resources='{"bob": 8}' --disable-usage-stats
```

### Connect to the cluster
Now you can start secretflow.
```python
>>> import secretflow as sf
>>> password=
>>> sf.init(address='ip:port', _redis_password=password)
>>> alice = sf.PYU('alice')
>>> bob = sf.PYU('bob')
>>> alice(lambda x : x)(2)
<secretflow.device.device.pyu.PYUObject object at 0x7fe932a1a640>
>>> bob(lambda x : x)(2)
<secretflow.device.device.pyu.PYUObject object at 0x7fe6fef03250>
```

### Suggestions for production
Secretflow use `ray` as its distribution system. 
You may need to do some more configuration for higher security when using it in production.
The following actions can help improve security features.

1. Enable tls Authentication.

    Ray can be configured to use TLS on its gRPC channels, for more, please refer to [Ray TLS](https://docs.ray.io/en/latest/ray-core/configure.html#tls-authentication).

2. Forbidden in-fly remote.

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