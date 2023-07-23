# Component Specification

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
    - [StorageConfig](#storageconfig)
    - [StorageConfig.LocalFSConfig](#storageconfiglocalfsconfig)





 ### data.proto



- Messages
    - [DeviceObjectCollection](#deviceobjectcollection)
    - [DeviceObjectCollection.DeviceObject](#deviceobjectcollectiondeviceobject)
    - [DistData](#distdata)
    - [DistData.DataRef](#distdatadataref)
    - [IndividualTable](#individualtable)
    - [SystemInfo](#systeminfo)
    - [TableSchema](#tableschema)
    - [VerticalTable](#verticaltable)





 ### comp.proto



- Messages
    - [Attribute](#attribute)
    - [AttributeDef](#attributedef)
    - [AttributeDef.AtomicAttrDesc](#attributedefatomicattrdesc)
    - [AttributeDef.UnionAttrGroupDesc](#attributedefunionattrgroupdesc)
    - [CompListDef](#complistdef)
    - [ComponentDef](#componentdef)
    - [IoDef](#iodef)
    - [IoDef.TableAttrDef](#iodeftableattrdef)



- Enums
    - [AttrType](#attrtype)




 ### evaluation.proto



- Messages
    - [NodeEvalParam](#nodeevalparam)
    - [NodeEvalResult](#nodeevalresult)





 ### report.proto



- Messages
    - [Descriptions](#descriptions)
    - [Descriptions.Item](#descriptionsitem)
    - [Div](#div)
    - [Div.Child](#divchild)
    - [Report](#report)
    - [Tab](#tab)
    - [Table](#table)
    - [Table.HeaderItem](#tableheaderitem)
    - [Table.Row](#tablerow)







 ## cluster.proto

Proto file: [secretflow/protos/component/cluster.proto](https://github.com/secretflow/secretflow/tree/main/secretflow/protos/component/cluster.proto)


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
| storage_config | [ StorageConfig](#storageconfig) | none |
 <!-- end Fields -->
 <!-- end HasFields -->


#### SFClusterConfig.PublicConfig
Public and shared to all parties.


| Field | Type | Description |
| ----- | ---- | ----------- |
| rayfed_config | [ SFClusterConfig.RayFedConfig](#sfclusterconfigrayfedconfig) | none |
| spu_configs | [repeated SFClusterConfig.SPUConfig](#sfclusterconfigspuconfig) | none |
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


| Field | Type | Description |
| ----- | ---- | ----------- |
| name | [ string](#string) | Name of the device. |
| type | [ string](#string) | Supported: SPU, HEU, TEEU. |
| parties | [repeated string](#string) | Parties of device. |
| config | [ string](#string) | Specific config for the secret device. |
 <!-- end Fields -->
 <!-- end HasFields -->


#### StorageConfig
A StorageConfig specifies the root for all data for one party.
- At this moment, only local_fs is supported. We would support OSS, database
in future.


| Field | Type | Description |
| ----- | ---- | ----------- |
| type | [ string](#string) | Supported: local_fs. |
| local_fs | [ StorageConfig.LocalFSConfig](#storageconfiglocalfsconfig) | local_fs config. |
 <!-- end Fields -->
 <!-- end HasFields -->


#### StorageConfig.LocalFSConfig



| Field | Type | Description |
| ----- | ---- | ----------- |
| wd | [ string](#string) | Working directory. |
 <!-- end Fields -->
 <!-- end HasFields -->
 <!-- end messages -->

### Enums
 <!-- end Enums -->


 ## data.proto

Proto file: [secretflow/protos/component/data.proto](https://github.com/secretflow/secretflow/tree/main/secretflow/protos/component/data.proto)


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


#### DistData
A public record for a general distributed data.

The type of this distributed data, should be meaningful to components.

The concrete data format (include public and private parts) is defined by
other protos.

Suggested names, i.e.
- sf.table.vertical_table      represent a secretflow vertical table
- sf.model.*                   represent a secretflow models.
- sf.rule.*                    represent a secretflow rules.


| Field | Type | Description |
| ----- | ---- | ----------- |
| name | [ string](#string) | The name of this distributed data. |
| type | [ string](#string) | none |
| sys_info | [ SystemInfo](#systeminfo) | Describe the system information that used to generate this distributed data. |
| meta | [ google.protobuf.Any](#googleprotobufany) | Public information, known to all parties. i.e. VerticalTable |
| data_refs | [repeated DistData.DataRef](#distdatadataref) | Remote data references. |
 <!-- end Fields -->
 <!-- end HasFields -->


#### DistData.DataRef
A reference to a data that is stored in the remote path.


| Field | Type | Description |
| ----- | ---- | ----------- |
| uri | [ string](#string) | The path information relative to StorageConfig of the party. |
| party | [ string](#string) | The owner party. |
| format | [ string](#string) | The storage format, i.e. csv. |
 <!-- end Fields -->
 <!-- end HasFields -->


#### IndividualTable
IndividualTable describes a table owned by a single party.


| Field | Type | Description |
| ----- | ---- | ----------- |
| schema | [ TableSchema](#tableschema) | none |
| num_lines | [ int64](#int64) | If -1, the number is unknown. |
 <!-- end Fields -->
 <!-- end HasFields -->


#### SystemInfo
Describe the application related to data.
- SCQL, GRM related meta information should be here.
- You can add more field here, when another application is added.


| Field | Type | Description |
| ----- | ---- | ----------- |
| app_name | [ string](#string) | The application name. Supported: `secretflow` |
| secretflow | [ SFClusterDesc](#sfclusterdesc) | For secretflow. |
 <!-- end Fields -->
 <!-- end HasFields -->


#### TableSchema
The schema of a table.
- A col must be one of `id | feature | label`. By default, it should be a
feature.
- All names must match the regexp `[A-Za-z0-9.][A-Za-z0-9_>./]*`.
- All data type must be one of
`component.data_utils.SUPPORTED_VTABLE_DATA_TYPE`, including
 `i8 | i16 | i32 | i64 | u8 | u16 | u32 | u64 | f16 | f32 | f64 | str`.


| Field | Type | Description |
| ----- | ---- | ----------- |
| ids | [repeated string](#string) | Id column name(s). Optional, can be empty. |
| features | [repeated string](#string) | Feature column name(s). |
| labels | [repeated string](#string) | Label column name(s). Optional, can be empty. |
| id_types | [repeated string](#string) | Id column data type(s). Len(id) should match len(id_types). |
| feature_types | [repeated string](#string) | Feature column data type(s). Len(features) should match len(feature_types). |
| label_types | [repeated string](#string) | Label column data type(s). Len(labels) should match len(label_types). |
 <!-- end Fields -->
 <!-- end HasFields -->


#### VerticalTable
VerticalTable describes a vertical virtual table from multiple parties.
> TODO: move this to secretflow/protos/builtin/

> Guide: if some type is only required to be handle inside a specific system,
for instance woe.rule file in engine, we don't need to define a new type
here.


| Field | Type | Description |
| ----- | ---- | ----------- |
| schemas | [repeated TableSchema](#tableschema) | The vertical partitioned slices' schema. Must match data_refs in the parent DistData message. |
| num_lines | [ int64](#int64) | If -1, the number is unknown. |
 <!-- end Fields -->
 <!-- end HasFields -->
 <!-- end messages -->

### Enums
 <!-- end Enums -->


 ## comp.proto

Proto file: [secretflow/protos/component/comp.proto](https://github.com/secretflow/secretflow/tree/main/secretflow/protos/component/comp.proto)


 <!-- end services -->

### Messages


#### Attribute
The value of an attribute


| Field | Type | Description |
| ----- | ---- | ----------- |
| f | [ float](#float) | FLOAT |
| i64 | [ int64](#int64) | INT NOTE(junfeng): "is" is preserved by Python. Replaced with "i64". |
| s | [ string](#string) | STRING |
| b | [ bool](#bool) | BOOL |
| fs | [repeated float](#float) | FLOATS |
| i64s | [repeated int64](#int64) | INTS |
| ss | [repeated string](#string) | STRINGS |
| bs | [repeated bool](#bool) | BOOLS |
| is_na | [ bool](#bool) | Indicates the value is missing explicitly. |
 <!-- end Fields -->
 <!-- end HasFields -->


#### AttributeDef
Describe an attribute.


| Field | Type | Description |
| ----- | ---- | ----------- |
| prefixes | [repeated string](#string) | Indicates the ancestors of a node, e.g. `[name_a, name_b, name_c]` means the path prefixes of current Attribute is `name_a/name_b/name_c/`. Only `^[a-zA-Z0-9_.-]*$` is allowed. `input` and `output` are reserved. |
| name | [ string](#string) | Must be unique in the same level just like Linux file systems. Only `^[a-zA-Z0-9_.-]*$` is allowed. `input` and `output` are reserved. |
| desc | [ string](#string) | none |
| type | [ AttrType](#attrtype) | none |
| atomic | [ AttributeDef.AtomicAttrDesc](#attributedefatomicattrdesc) | none |
| union | [ AttributeDef.UnionAttrGroupDesc](#attributedefunionattrgroupdesc) | none |
 <!-- end Fields -->
 <!-- end HasFields -->


#### AttributeDef.AtomicAttrDesc
Extras for an atomic attribute.
Including: `AT_FLOAT | AT_INT | AT_STRING | AT_BOOL | AT_FLOATS | AT_INTS |
AT_STRINGS | AT_BOOLS`.


| Field | Type | Description |
| ----- | ---- | ----------- |
| list_min_length_inclusive | [ int64](#int64) | Only valid when type is `AT_FLOATS \| AT_INTS \| AT_STRINGS \| AT_BOOLS`. |
| list_max_length_inclusive | [ int64](#int64) | Only valid when type is `AT_FLOATS \| AT_INTS \| AT_STRINGS \| AT_BOOLS`. |
| is_optional | [ bool](#bool) | none |
| default_value | [ Attribute](#attribute) | A reasonable default for this attribute if the user does not supply a value. |
| allowed_values | [ Attribute](#attribute) | Only valid when type is `AT_FLOAT \| AT_INT \| AT_STRING \| AT_FLOATS \| AT_INTS \| AT_STRINGS`. Please use list fields of AtomicParameter, i.e. `ss`, `i64s`, `fs`. If the attribute is a list, allowed_values is applied to each element. |
| has_lower_bound | [ bool](#bool) | Only valid when type is `AT_FLOAT \| AT_INT \| AT_FLOATS \| AT_INTS `. If the attribute is a list, lower_bound is applied to each element. |
| lower_bound | [ Attribute](#attribute) | none |
| lower_bound_inclusive | [ bool](#bool) | none |
| has_upper_bound | [ bool](#bool) | Only valid when type is `AT_FLOAT \| AT_INT \| AT_FLOATS \| AT_INTS `. If the attribute is a list, upper_bound is applied to each element. |
| upper_bound | [ Attribute](#attribute) | none |
| upper_bound_inclusive | [ bool](#bool) | none |
 <!-- end Fields -->
 <!-- end HasFields -->


#### AttributeDef.UnionAttrGroupDesc
Extras for a union attribute group.


| Field | Type | Description |
| ----- | ---- | ----------- |
| default_selection | [ string](#string) | The default selected child. |
 <!-- end Fields -->
 <!-- end HasFields -->


#### CompListDef
A list of components


| Field | Type | Description |
| ----- | ---- | ----------- |
| name | [ string](#string) | none |
| desc | [ string](#string) | none |
| version | [ string](#string) | none |
| comps | [repeated ComponentDef](#componentdef) | none |
 <!-- end Fields -->
 <!-- end HasFields -->


#### ComponentDef
The definition of a comp.


| Field | Type | Description |
| ----- | ---- | ----------- |
| domain | [ string](#string) | Namespace of the comp. |
| name | [ string](#string) | Should be unique among all comps of the same domain. |
| desc | [ string](#string) | none |
| version | [ string](#string) | Version of the comp. |
| attrs | [repeated AttributeDef](#attributedef) | none |
| inputs | [repeated IoDef](#iodef) | none |
| outputs | [repeated IoDef](#iodef) | none |
 <!-- end Fields -->
 <!-- end HasFields -->


#### IoDef
Define an input/output for component.


| Field | Type | Description |
| ----- | ---- | ----------- |
| name | [ string](#string) | should be unique among all IOs of the component. |
| desc | [ string](#string) | none |
| types | [repeated string](#string) | Must be one of DistData.type in data.proto |
| attrs | [repeated IoDef.TableAttrDef](#iodeftableattrdef) | Only valid for tables. The attribute path for a TableAttrDef is `{input\|output}/{IoDef name}/{TableAttrDef name}`. |
 <!-- end Fields -->
 <!-- end HasFields -->


#### IoDef.TableAttrDef
An extra attribute for a table.

If provided in a IoDef, e.g.
```json
{
  "name": "feature",
  "types": [
      "int",
      "float"
  ],
  "col_min_cnt_inclusive": 1,
  "col_max_cnt": 3,
  "attrs": [
      {
          "name": "bucket_size",
          "type": "AT_INT"
      }
  ]
}
```
means after a user provide a table as IO, they should also specify
cols as "feature":
- col_min_cnt_inclusive is 1: At least 1 col to be selected.
- col_max_cnt_inclusive is 3: At most 3 cols to be selected.
And afterwards, user have to fill an int attribute called bucket_size for
each selected cols.


| Field | Type | Description |
| ----- | ---- | ----------- |
| name | [ string](#string) | Must be unique among all attributes for the table. |
| desc | [ string](#string) | none |
| types | [repeated string](#string) | Accepted col data types. Please check DistData.VerticalTable in data.proto. |
| col_min_cnt_inclusive | [ int64](#int64) | inclusive |
| col_max_cnt_inclusive | [ int64](#int64) | none |
| attrs | [repeated AttributeDef](#attributedef) | extra attribute for specified col. |
 <!-- end Fields -->
 <!-- end HasFields -->
 <!-- end messages -->

### Enums


#### AttrType
Supported attribute types.

| Name | Number | Description |
| ---- | ------ | ----------- |
| AT_UNDEFINED | 0 | none |
| AT_FLOAT | 1 | FLOAT |
| AT_INT | 2 | INT |
| AT_STRING | 3 | STRING |
| AT_BOOL | 4 | BOOL |
| AT_FLOATS | 5 | FLOATS |
| AT_INTS | 6 | INTS |
| AT_STRINGS | 7 | STRINGS |
| AT_BOOLS | 8 | BOOLS |
| AT_STRUCT_GROUP | 9 | none |
| AT_UNION_GROUP | 10 | none |
| AT_SF_TABLE_COL | 11 | none |


 <!-- end Enums -->


 ## evaluation.proto

Proto file: [secretflow/protos/component/evaluation.proto](https://github.com/secretflow/secretflow/tree/main/secretflow/protos/component/evaluation.proto)


 <!-- end services -->

### Messages


#### NodeEvalParam
Evaluate a node.
- comp.evaluate(NodeEvalParam, SFClusterConfig) -> NodeEvalResult

NodeEvalParam contains all the information to evaluate a component.


| Field | Type | Description |
| ----- | ---- | ----------- |
| domain | [ string](#string) | Domain of the component. |
| name | [ string](#string) | Name of the component. |
| version | [ string](#string) | Version of the component. |
| attr_paths | [repeated string](#string) | The path of attributes. The attribute path for a TableAttrDef is `{input\|output}/{IoDef name}/{TableAttrDef name}`. |
| attrs | [repeated Attribute](#attribute) | The value of the attribute. Must match attr_paths. |
| inputs | [repeated DistData](#distdata) | The input data, the order of inputs must match inputs in ComponentDef. NOTE: Names of DistData doesn't need to match those of inputs in ComponentDef definition. |
| output_uris | [repeated string](#string) | The output data uris, the order of output_uris must match outputs in ComponentDef. |
 <!-- end Fields -->
 <!-- end HasFields -->


#### NodeEvalResult
NodeEvalResult contains outputs of a component evaluation.


| Field | Type | Description |
| ----- | ---- | ----------- |
| outputs | [repeated DistData](#distdata) | Output data. |
 <!-- end Fields -->
 <!-- end HasFields -->
 <!-- end messages -->

### Enums
 <!-- end Enums -->


 ## report.proto

Proto file: [secretflow/protos/component/report.proto](https://github.com/secretflow/secretflow/tree/main/secretflow/protos/component/report.proto)


 <!-- end services -->

### Messages


#### Descriptions
Displays multiple read-only fields in groups.


| Field | Type | Description |
| ----- | ---- | ----------- |
| name | [ string](#string) | Name of the Descriptions. |
| desc | [ string](#string) | none |
| items | [repeated Descriptions.Item](#descriptionsitem) | none |
 <!-- end Fields -->
 <!-- end HasFields -->


#### Descriptions.Item



| Field | Type | Description |
| ----- | ---- | ----------- |
| name | [ string](#string) | Name of the field. |
| desc | [ string](#string) | none |
| type | [ AttrType](#attrtype) | none |
| value | [ Attribute](#attribute) | none |
 <!-- end Fields -->
 <!-- end HasFields -->


#### Div
A division or a section of a page.


| Field | Type | Description |
| ----- | ---- | ----------- |
| name | [ string](#string) | Name of the Div. |
| desc | [ string](#string) | none |
| children | [repeated Div.Child](#divchild) | none |
 <!-- end Fields -->
 <!-- end HasFields -->


#### Div.Child



| Field | Type | Description |
| ----- | ---- | ----------- |
| type | [ string](#string) | Supported: descriptions, table, div. |
| descriptions | [ Descriptions](#descriptions) | none |
| table | [ Table](#table) | none |
| div | [ Div](#div) | none |
 <!-- end Fields -->
 <!-- end HasFields -->


#### Report



| Field | Type | Description |
| ----- | ---- | ----------- |
| name | [ string](#string) | Name of the Report. |
| desc | [ string](#string) | none |
| tabs | [repeated Tab](#tab) | none |
| err_code | [ int32](#int32) | none |
| err_detail | [ string](#string) | Structed error detail (JSON encoded message). |
 <!-- end Fields -->
 <!-- end HasFields -->


#### Tab
A page of a report.


| Field | Type | Description |
| ----- | ---- | ----------- |
| name | [ string](#string) | Name of the Tab. |
| desc | [ string](#string) | none |
| divs | [repeated Div](#div) | none |
 <!-- end Fields -->
 <!-- end HasFields -->


#### Table
Displays rows of data.


| Field | Type | Description |
| ----- | ---- | ----------- |
| name | [ string](#string) | Name of the Table. |
| desc | [ string](#string) | none |
| headers | [repeated Table.HeaderItem](#tableheaderitem) | none |
| rows | [repeated Table.Row](#tablerow) | none |
 <!-- end Fields -->
 <!-- end HasFields -->


#### Table.HeaderItem



| Field | Type | Description |
| ----- | ---- | ----------- |
| name | [ string](#string) | none |
| desc | [ string](#string) | none |
| type | [ AttrType](#attrtype) | none |
 <!-- end Fields -->
 <!-- end HasFields -->


#### Table.Row



| Field | Type | Description |
| ----- | ---- | ----------- |
| name | [ string](#string) | none |
| desc | [ string](#string) | none |
| items | [repeated Attribute](#attribute) | none |
 <!-- end Fields -->
 <!-- end HasFields -->
 <!-- end messages -->

### Enums
 <!-- end Enums -->
 <!-- end Files -->
