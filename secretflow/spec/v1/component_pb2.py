# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: secretflow/spec/v1/component.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\"secretflow/spec/v1/component.proto\x12\x12secretflow.spec.v1\"z\n\tAttribute\x12\t\n\x01\x66\x18\x01 \x01(\x02\x12\x0b\n\x03i64\x18\x02 \x01(\x03\x12\t\n\x01s\x18\x03 \x01(\t\x12\t\n\x01\x62\x18\x04 \x01(\x08\x12\n\n\x02\x66s\x18\x05 \x03(\x02\x12\x0c\n\x04i64s\x18\x06 \x03(\x03\x12\n\n\x02ss\x18\x07 \x03(\t\x12\n\n\x02\x62s\x18\x08 \x03(\x08\x12\r\n\x05is_na\x18\t \x01(\x08\"\xd9\x05\n\x0c\x41ttributeDef\x12\x10\n\x08prefixes\x18\x01 \x03(\t\x12\x0c\n\x04name\x18\x02 \x01(\t\x12\x0c\n\x04\x64\x65sc\x18\x03 \x01(\t\x12*\n\x04type\x18\x04 \x01(\x0e\x32\x1c.secretflow.spec.v1.AttrType\x12?\n\x06\x61tomic\x18\x05 \x01(\x0b\x32/.secretflow.spec.v1.AttributeDef.AtomicAttrDesc\x12\x42\n\x05union\x18\x06 \x01(\x0b\x32\x33.secretflow.spec.v1.AttributeDef.UnionAttrGroupDesc\x1a\xb8\x03\n\x0e\x41tomicAttrDesc\x12!\n\x19list_min_length_inclusive\x18\x01 \x01(\x03\x12!\n\x19list_max_length_inclusive\x18\x02 \x01(\x03\x12\x13\n\x0bis_optional\x18\x03 \x01(\x08\x12\x34\n\rdefault_value\x18\x04 \x01(\x0b\x32\x1d.secretflow.spec.v1.Attribute\x12\x35\n\x0e\x61llowed_values\x18\x05 \x01(\x0b\x32\x1d.secretflow.spec.v1.Attribute\x12\x1b\n\x13lower_bound_enabled\x18\x06 \x01(\x08\x12\x32\n\x0blower_bound\x18\x07 \x01(\x0b\x32\x1d.secretflow.spec.v1.Attribute\x12\x1d\n\x15lower_bound_inclusive\x18\x08 \x01(\x08\x12\x1b\n\x13upper_bound_enabled\x18\t \x01(\x08\x12\x32\n\x0bupper_bound\x18\n \x01(\x0b\x32\x1d.secretflow.spec.v1.Attribute\x12\x1d\n\x15upper_bound_inclusive\x18\x0b \x01(\x08\x1a/\n\x12UnionAttrGroupDesc\x12\x19\n\x11\x64\x65\x66\x61ult_selection\x18\x01 \x01(\t\"\x9a\x02\n\x05IoDef\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0c\n\x04\x64\x65sc\x18\x02 \x01(\t\x12\r\n\x05types\x18\x03 \x03(\t\x12\x35\n\x05\x61ttrs\x18\x04 \x03(\x0b\x32&.secretflow.spec.v1.IoDef.TableAttrDef\x1a\xae\x01\n\x0cTableAttrDef\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0c\n\x04\x64\x65sc\x18\x02 \x01(\t\x12\r\n\x05types\x18\x03 \x03(\t\x12\x1d\n\x15\x63ol_min_cnt_inclusive\x18\x04 \x01(\x03\x12\x1d\n\x15\x63ol_max_cnt_inclusive\x18\x05 \x01(\x03\x12\x35\n\x0b\x65xtra_attrs\x18\x06 \x03(\x0b\x32 .secretflow.spec.v1.AttributeDef\"\xd3\x01\n\x0c\x43omponentDef\x12\x0e\n\x06\x64omain\x18\x01 \x01(\t\x12\x0c\n\x04name\x18\x02 \x01(\t\x12\x0c\n\x04\x64\x65sc\x18\x03 \x01(\t\x12\x0f\n\x07version\x18\x04 \x01(\t\x12/\n\x05\x61ttrs\x18\x05 \x03(\x0b\x32 .secretflow.spec.v1.AttributeDef\x12)\n\x06inputs\x18\x06 \x03(\x0b\x32\x19.secretflow.spec.v1.IoDef\x12*\n\x07outputs\x18\x07 \x03(\x0b\x32\x19.secretflow.spec.v1.IoDef\"k\n\x0b\x43ompListDef\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0c\n\x04\x64\x65sc\x18\x02 \x01(\t\x12\x0f\n\x07version\x18\x03 \x01(\t\x12/\n\x05\x63omps\x18\x04 \x03(\x0b\x32 .secretflow.spec.v1.ComponentDef*\xd3\x01\n\x08\x41ttrType\x12\x19\n\x15\x41TTR_TYPE_UNSPECIFIED\x10\x00\x12\x0c\n\x08\x41T_FLOAT\x10\x01\x12\n\n\x06\x41T_INT\x10\x02\x12\r\n\tAT_STRING\x10\x03\x12\x0b\n\x07\x41T_BOOL\x10\x04\x12\r\n\tAT_FLOATS\x10\x05\x12\x0b\n\x07\x41T_INTS\x10\x06\x12\x0e\n\nAT_STRINGS\x10\x07\x12\x0c\n\x08\x41T_BOOLS\x10\x08\x12\x13\n\x0f\x41T_STRUCT_GROUP\x10\t\x12\x12\n\x0e\x41T_UNION_GROUP\x10\n\x12\x13\n\x0f\x41T_SF_TABLE_COL\x10\x0b\x42*\n\x16\x63om.secretflow.spec.v1B\x0e\x43omponentProtoP\x01\x62\x06proto3')

_ATTRTYPE = DESCRIPTOR.enum_types_by_name['AttrType']
AttrType = enum_type_wrapper.EnumTypeWrapper(_ATTRTYPE)
ATTR_TYPE_UNSPECIFIED = 0
AT_FLOAT = 1
AT_INT = 2
AT_STRING = 3
AT_BOOL = 4
AT_FLOATS = 5
AT_INTS = 6
AT_STRINGS = 7
AT_BOOLS = 8
AT_STRUCT_GROUP = 9
AT_UNION_GROUP = 10
AT_SF_TABLE_COL = 11


_ATTRIBUTE = DESCRIPTOR.message_types_by_name['Attribute']
_ATTRIBUTEDEF = DESCRIPTOR.message_types_by_name['AttributeDef']
_ATTRIBUTEDEF_ATOMICATTRDESC = _ATTRIBUTEDEF.nested_types_by_name['AtomicAttrDesc']
_ATTRIBUTEDEF_UNIONATTRGROUPDESC = _ATTRIBUTEDEF.nested_types_by_name['UnionAttrGroupDesc']
_IODEF = DESCRIPTOR.message_types_by_name['IoDef']
_IODEF_TABLEATTRDEF = _IODEF.nested_types_by_name['TableAttrDef']
_COMPONENTDEF = DESCRIPTOR.message_types_by_name['ComponentDef']
_COMPLISTDEF = DESCRIPTOR.message_types_by_name['CompListDef']
Attribute = _reflection.GeneratedProtocolMessageType('Attribute', (_message.Message,), {
  'DESCRIPTOR' : _ATTRIBUTE,
  '__module__' : 'secretflow.spec.v1.component_pb2'
  # @@protoc_insertion_point(class_scope:secretflow.spec.v1.Attribute)
  })
_sym_db.RegisterMessage(Attribute)

AttributeDef = _reflection.GeneratedProtocolMessageType('AttributeDef', (_message.Message,), {

  'AtomicAttrDesc' : _reflection.GeneratedProtocolMessageType('AtomicAttrDesc', (_message.Message,), {
    'DESCRIPTOR' : _ATTRIBUTEDEF_ATOMICATTRDESC,
    '__module__' : 'secretflow.spec.v1.component_pb2'
    # @@protoc_insertion_point(class_scope:secretflow.spec.v1.AttributeDef.AtomicAttrDesc)
    })
  ,

  'UnionAttrGroupDesc' : _reflection.GeneratedProtocolMessageType('UnionAttrGroupDesc', (_message.Message,), {
    'DESCRIPTOR' : _ATTRIBUTEDEF_UNIONATTRGROUPDESC,
    '__module__' : 'secretflow.spec.v1.component_pb2'
    # @@protoc_insertion_point(class_scope:secretflow.spec.v1.AttributeDef.UnionAttrGroupDesc)
    })
  ,
  'DESCRIPTOR' : _ATTRIBUTEDEF,
  '__module__' : 'secretflow.spec.v1.component_pb2'
  # @@protoc_insertion_point(class_scope:secretflow.spec.v1.AttributeDef)
  })
_sym_db.RegisterMessage(AttributeDef)
_sym_db.RegisterMessage(AttributeDef.AtomicAttrDesc)
_sym_db.RegisterMessage(AttributeDef.UnionAttrGroupDesc)

IoDef = _reflection.GeneratedProtocolMessageType('IoDef', (_message.Message,), {

  'TableAttrDef' : _reflection.GeneratedProtocolMessageType('TableAttrDef', (_message.Message,), {
    'DESCRIPTOR' : _IODEF_TABLEATTRDEF,
    '__module__' : 'secretflow.spec.v1.component_pb2'
    # @@protoc_insertion_point(class_scope:secretflow.spec.v1.IoDef.TableAttrDef)
    })
  ,
  'DESCRIPTOR' : _IODEF,
  '__module__' : 'secretflow.spec.v1.component_pb2'
  # @@protoc_insertion_point(class_scope:secretflow.spec.v1.IoDef)
  })
_sym_db.RegisterMessage(IoDef)
_sym_db.RegisterMessage(IoDef.TableAttrDef)

ComponentDef = _reflection.GeneratedProtocolMessageType('ComponentDef', (_message.Message,), {
  'DESCRIPTOR' : _COMPONENTDEF,
  '__module__' : 'secretflow.spec.v1.component_pb2'
  # @@protoc_insertion_point(class_scope:secretflow.spec.v1.ComponentDef)
  })
_sym_db.RegisterMessage(ComponentDef)

CompListDef = _reflection.GeneratedProtocolMessageType('CompListDef', (_message.Message,), {
  'DESCRIPTOR' : _COMPLISTDEF,
  '__module__' : 'secretflow.spec.v1.component_pb2'
  # @@protoc_insertion_point(class_scope:secretflow.spec.v1.CompListDef)
  })
_sym_db.RegisterMessage(CompListDef)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\n\026com.secretflow.spec.v1B\016ComponentProtoP\001'
  _ATTRTYPE._serialized_start=1523
  _ATTRTYPE._serialized_end=1734
  _ATTRIBUTE._serialized_start=58
  _ATTRIBUTE._serialized_end=180
  _ATTRIBUTEDEF._serialized_start=183
  _ATTRIBUTEDEF._serialized_end=912
  _ATTRIBUTEDEF_ATOMICATTRDESC._serialized_start=423
  _ATTRIBUTEDEF_ATOMICATTRDESC._serialized_end=863
  _ATTRIBUTEDEF_UNIONATTRGROUPDESC._serialized_start=865
  _ATTRIBUTEDEF_UNIONATTRGROUPDESC._serialized_end=912
  _IODEF._serialized_start=915
  _IODEF._serialized_end=1197
  _IODEF_TABLEATTRDEF._serialized_start=1023
  _IODEF_TABLEATTRDEF._serialized_end=1197
  _COMPONENTDEF._serialized_start=1200
  _COMPONENTDEF._serialized_end=1411
  _COMPLISTDEF._serialized_start=1413
  _COMPLISTDEF._serialized_end=1520
# @@protoc_insertion_point(module_scope)