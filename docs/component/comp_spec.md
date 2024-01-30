# Extended Specification

## Table of Contents


 ### cluster.proto



- Messages
    - [SFClusterConfig](#sfclusterconfig)
    - [SFClusterConfig.PrivateConfig](#sfclusterconfigprivateconfig)
    - [SFClusterConfig.PublicConfig](#sfclusterconfigpublicconfig)
    - [SFClusterConfig.RayFedConfig](#sfclusterconfigrayfedconfig)
    - [SFClusterConfig.SPUConfig](#sfclusterconfigspuconfig)
    - [SFClusterDesc](#sfclusterdesc)
    - [SFClusterDesc.DeviceDesc](#sfclusterdescdevicedesc)
    - [SFClusterDesc.RayFedConfig](#sfclusterdescrayfedconfig)
  




 ### data.proto



- Messages
    - [DeviceObjectCollection](#deviceobjectcollection)
    - [DeviceObjectCollection.DeviceObject](#deviceobjectcollectiondeviceobject)
  






 ## cluster.proto

Proto file: [secretflow/protos/secretflow/spec/extend/cluster.proto](https://github.com/secretflow/secretflow/tree/main/secretflow/protos/secretflow/spec/extend/cluster.proto)


 <!-- end services -->

### Messages


#### SFClusterConfig
Runtime Config for a SecretFlow cluster.
Besides intrinsic SFClusterDesc, dynamic network configs are provided.


| Field | Type | Description |
| ----- | ---- | ----------- |
| desc | [ SFClusterDesc](#sfclusterdesc) | Intrinsic properties. |
| public_config | [ SFClusterConfig.PublicConfig](#sfclusterconfigpublicconfig) | Dynamic runtime public configs. |
| private_config | [ SFClusterConfig.PrivateConfig](#sfclusterconfigprivateconfig) | Dynamic runtime private configs. |
 <!-- end Fields -->
 <!-- end HasFields -->


#### SFClusterConfig.PrivateConfig
Different for each party.
Private and unique to each party.


| Field | Type | Description |
| ----- | ---- | ----------- |
| self_party | [ string](#string) | none |
| ray_head_addr | [ string](#string) | none |
 <!-- end Fields -->
 <!-- end HasFields -->


#### SFClusterConfig.PublicConfig
Public and shared to all parties.


| Field | Type | Description |
| ----- | ---- | ----------- |
| ray_fed_config | [ SFClusterConfig.RayFedConfig](#sfclusterconfigrayfedconfig) | none |
| spu_configs | [repeated SFClusterConfig.SPUConfig](#sfclusterconfigspuconfig) | none |
| barrier_on_shutdown | [ bool](#bool) | none |
 <!-- end Fields -->
 <!-- end HasFields -->


#### SFClusterConfig.RayFedConfig
Addresses for the RayFed cluster of each party.


| Field | Type | Description |
| ----- | ---- | ----------- |
| parties | [repeated string](#string) | none |
| addresses | [repeated string](#string) | none |
| listen_addresses | [repeated string](#string) | none |
 <!-- end Fields -->
 <!-- end HasFields -->


#### SFClusterConfig.SPUConfig
Contains addresses for one SPU device.


| Field | Type | Description |
| ----- | ---- | ----------- |
| name | [ string](#string) | Should match SPU name in SFClusterDesc.devices. |
| parties | [repeated string](#string) | none |
| addresses | [repeated string](#string) | none |
| listen_addresses | [repeated string](#string) | none |
 <!-- end Fields -->
 <!-- end HasFields -->


#### SFClusterDesc
Intrinsic properties of a SecretFlow cluster, including:

- Version info.
- Parties: who participate in the computation.
- Secret devices including  and their configs.


| Field | Type | Description |
| ----- | ---- | ----------- |
| sf_version | [ string](#string) | SecretFlow version. |
| py_version | [ string](#string) | Python version. |
| parties | [repeated string](#string) | Joined entities. e.g. ["alice", "bob",...]. |
| devices | [repeated SFClusterDesc.DeviceDesc](#sfclusterdescdevicedesc) | Description of secret devices |
| ray_fed_config | [ SFClusterDesc.RayFedConfig](#sfclusterdescrayfedconfig) | none |
 <!-- end Fields -->
 <!-- end HasFields -->


#### SFClusterDesc.DeviceDesc
Description for a secret device.
- PYUs do not need to claim since they are plaintext devices.
- Notes for config:
At this moment, you have to provide a JSON string for different devices.
We are going to formalize this part in future.
  * Example SPU config:

```json
  {
    "runtime_config": {
        "protocol": "REF2K",
        "field": "FM64"
    },
    "link_desc": {
        "connect_retry_times": 60,
        "connect_retry_interval_ms": 1000,
        "brpc_channel_protocol": "http",
        "brpc_channel_connection_type": "pooled",
        "recv_timeout_ms": 1200000,
        "http_timeout_ms": 1200000
    }
}
```
Referrences:
SPU:
https://www.secretflow.org.cn/docs/spu/latest/en-US/reference/runtime_config#runtimeconfig
HEU:
https://www.secretflow.org.cn/docs/secretflow/latest/en-US/source/secretflow.device.device.device#secretflow.device.device.heu.HEU.__init__


| Field | Type | Description |
| ----- | ---- | ----------- |
| name | [ string](#string) | Name of the device. |
| type | [ string](#string) | Supported: SPU, HEU, TEEU. |
| parties | [repeated string](#string) | Parties of device. |
| config | [ string](#string) | Specific config for the secret device. |
 <!-- end Fields -->
 <!-- end HasFields -->


#### SFClusterDesc.RayFedConfig



| Field | Type | Description |
| ----- | ---- | ----------- |
| cross_silo_comm_backend | [ string](#string) | Indicates communication backend of RayFed. Accepted: 'grpc', 'brpc_link' Dafault is 'grpc' |
 <!-- end Fields -->
 <!-- end HasFields -->
 <!-- end messages -->

### Enums
 <!-- end Enums -->


 ## data.proto

Proto file: [secretflow/protos/secretflow/spec/extend/data.proto](https://github.com/secretflow/secretflow/tree/main/secretflow/protos/secretflow/spec/extend/data.proto)


 <!-- end services -->

### Messages


#### DeviceObjectCollection
Descibes public storage info for a collection of Device Objects.


| Field | Type | Description |
| ----- | ---- | ----------- |
| objs | [repeated DeviceObjectCollection.DeviceObject](#deviceobjectcollectiondeviceobject) | none |
| public_info | [ string](#string) | Any public information. |
 <!-- end Fields -->
 <!-- end HasFields -->


#### DeviceObjectCollection.DeviceObject



| Field | Type | Description |
| ----- | ---- | ----------- |
| type | [ string](#string) | Supported: `spu \| pyu` |
| data_ref_idxs | [repeated int32](#int32) | Index of data_ref in the parent DistData message. |
 <!-- end Fields -->
 <!-- end HasFields -->
 <!-- end messages -->

### Enums
 <!-- end Enums -->
 <!-- end Files -->
