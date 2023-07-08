#
# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from secretflow.component.entry import COMP_LIST
from secretflow.protos.component.comp_pb2 import AttrType, Attribute
from mdutils.mdutils import MdUtils

import datetime

this_directory = os.path.abspath(os.path.dirname(__file__))

mdFile = MdUtils(
    file_name=os.path.join(this_directory, 'comp_list.md'),
)

mdFile.new_header(level=1, title='SecretFlow Component List', style='setext')

mdFile.new_paragraph(f'Last update: {datetime.datetime.now().strftime("%c")}')
mdFile.new_paragraph(f'Version: {COMP_LIST.version}')
mdFile.new_paragraph(COMP_LIST.desc)

AttrTypeStrMap = {
    AttrType.AT_UNDEFINED: 'Undefined',
    AttrType.AT_FLOAT: 'Float',
    AttrType.AT_INT: 'Integer',
    AttrType.AT_STRING: 'String',
    AttrType.AT_BOOL: 'Boolean',
    AttrType.AT_FLOATS: 'Float List',
    AttrType.AT_INTS: 'Integer List',
    AttrType.AT_STRINGS: 'String List',
    AttrType.AT_BOOLS: 'Boolean List',
    AttrType.AT_STRUCT_GROUP: 'Special type. Struct group. You must fill in all children.',
    AttrType.AT_UNION_GROUP: 'Special type. Union group. You must select one children to fill in.',
    AttrType.AT_SF_TABLE_COL: 'Special type. SecretFlow table column name.',
}


def get_atomic_attr_value(at: AttrType, attr: Attribute):
    if at == AttrType.AT_FLOAT:
        return round(attr.f, 5)
    elif at == AttrType.AT_INT:
        return attr.i64
    elif at == AttrType.AT_STRING:
        return attr.s
    elif at == AttrType.AT_BOOL:
        return attr.b
    elif at == AttrType.AT_FLOATS:
        return [round(f, 5) for f in attr.fs]
    elif at == AttrType.AT_INTS:
        return list(attr.i64s)
    elif at == AttrType.AT_STRINGS:
        return list(attr.ss)
    elif at == AttrType.AT_BOOLS:
        return list(attr.bs)
    else:
        return None


def get_allowed_atomic_attr_value(at: AttrType, attr: Attribute):
    if at == AttrType.AT_FLOAT or at == AttrType.AT_FLOATS:
        return [round(f, 5) for f in attr.fs]
    elif at == AttrType.AT_INT or at == AttrType.AT_INTS:
        return list(attr.i64s)
    elif at == AttrType.AT_STRING or at == AttrType.AT_STRINGS:
        return list(attr.ss)
    else:
        return None


def get_bound(
    at: AttrType,
    has_lower_bound: bool,
    lower_bound: Attribute,
    lower_bound_inclusive: bool,
    has_upper_bound: bool,
    upper_bound: Attribute,
    upper_bound_inclusive: bool,
):
    if at in [AttrType.AT_FLOAT, AttrType.AT_FLOATS, AttrType.AT_INT, AttrType.AT_INTS]:
        if has_lower_bound or has_upper_bound:
            ret = ''
            if has_lower_bound:
                ret += '[' if lower_bound_inclusive else '('
                ret += str(get_atomic_attr_value(at, lower_bound))
                ret += ', '
            else:
                ret += '($-\infty$, '

            if has_upper_bound:
                ret += str(get_atomic_attr_value(at, upper_bound))
                ret += ']' if upper_bound_inclusive else ')'
            else:
                ret += '$\infty$)'

            return ret

        else:
            return None
    else:
        return None


def parse_comp_io(md, io_defs):
    io_table_text = ['Name', 'Description', 'Type(s)', 'Notes']
    for io_def in io_defs:
        notes_str = ''
        if len(io_def.attrs):
            notes_str += "Extra table attributes."
            for i, attr in enumerate(list(io_def.attrs)):
                notes_str += f'({i}) {attr.name} - {attr.desc} '
                if len(attr.types):
                    notes_str += f"Accepted column types: {list(attr.types)}"

                if attr.col_min_cnt_inclusive > 0:
                    notes_str += f'Min column number to select(inclusive): {attr.col_min_cnt_inclusive}. '
                if attr.col_max_cnt_inclusive > 0:
                    notes_str += f'Max column number to select(inclusive): {attr.col_max_cnt_inclusive}. '

                if len(attr.attrs):
                    raise NotImplementedError('todo: parse attrs of TableAttrDef.')

        io_table_text.extend(
            [io_def.name, io_def.desc, str(list(io_def.types)), notes_str]
        )

    md.new_line()
    md.new_table(
        columns=4,
        rows=len(io_defs) + 1,
        text=io_table_text,
        text_align='left',
    )


comp_map = {}

for comp in COMP_LIST.comps:
    if comp.domain not in comp_map:
        comp_map[comp.domain] = {}
    comp_map[comp.domain][comp.name] = comp


for domain, comps in comp_map.items():
    mdFile.new_header(
        level=2,
        title=domain,
    )

    for name, comp_def in comps.items():
        mdFile.new_header(
            level=3,
            title=name,
        )
        mdFile.new_paragraph(f'Component version: {comp_def.version}')
        mdFile.new_paragraph(comp_def.desc)

        if len(comp_def.attrs):
            mdFile.new_header(
                level=4,
                title='Attrs',
            )
            attr_table_text = ["Name", "Description", "Type", "Required", "Notes"]
            for attr in comp_def.attrs:
                name_str = '/'.join(list(attr.prefixes) + [attr.name])
                type_str = AttrTypeStrMap[attr.type]
                required_str = 'N/A'
                notes_str = ''

                # atomic
                if attr.type in [
                    AttrType.AT_FLOAT,
                    AttrType.AT_INT,
                    AttrType.AT_STRING,
                    AttrType.AT_BOOL,
                    AttrType.AT_FLOATS,
                    AttrType.AT_INTS,
                    AttrType.AT_STRINGS,
                    AttrType.AT_BOOLS,
                ]:
                    if attr.type in [
                        AttrType.AT_FLOATS,
                        AttrType.AT_INTS,
                        AttrType.AT_STRINGS,
                        AttrType.AT_BOOLS,
                    ]:
                        if attr.atomic.list_min_length_inclusive > 0:
                            notes_str += f'Min length(inclusive): {attr.atomic.list_min_length_inclusive}. '
                        if attr.atomic.list_max_length_inclusive > 0:
                            notes_str += f'Max length(inclusive): {attr.atomic.list_max_length_inclusive}. '

                    default_value = get_atomic_attr_value(
                        attr.type, attr.atomic.default_value
                    )
                    if default_value is not None:
                        notes_str += f'Default: {default_value}. '

                    allowed_value = get_allowed_atomic_attr_value(
                        attr.type, attr.atomic.allowed_values
                    )
                    if allowed_value is not None and len(allowed_value):
                        notes_str += f'Allowed: {allowed_value}. '

                    required_str = 'N' if attr.atomic.is_optional else 'Y'

                    bound = get_bound(
                        attr.type,
                        attr.atomic.has_lower_bound,
                        attr.atomic.lower_bound,
                        attr.atomic.lower_bound_inclusive,
                        attr.atomic.has_upper_bound,
                        attr.atomic.upper_bound,
                        attr.atomic.upper_bound_inclusive,
                    )
                    if bound is not None:
                        notes_str += f'Range: {bound}. '
                else:
                    raise NotImplementedError('todo: parse other attr types.')

                attr_table_text.extend(
                    [name_str, attr.desc, type_str, required_str, notes_str.rstrip()]
                )

            mdFile.new_line()
            mdFile.new_table(
                columns=5,
                rows=len(comp_def.attrs) + 1,
                text=attr_table_text,
                text_align='left',
            )

        if len(comp_def.inputs):
            mdFile.new_header(
                level=4,
                title='Inputs',
            )

            parse_comp_io(mdFile, comp_def.inputs)
        if len(comp_def.outputs):
            mdFile.new_header(
                level=4,
                title='Outputs',
            )
            parse_comp_io(mdFile, comp_def.outputs)


mdFile.create_md_file()
