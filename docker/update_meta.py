import json
import logging
import os

import translators as ts
from google.protobuf.json_format import MessageToJson

from secretflow.component.entry import COMP_LIST
from secretflow.component.i18n import gettext

LANG = "zh"
TRANSLATOR = "bing"
this_directory = os.path.abspath(os.path.dirname(__file__))
COMP_LIST_FILE = os.path.join(this_directory, 'comp_list.json')
TRANSLATION_FILE = os.path.join(this_directory, 'translation.json')


def translate(input):
    output = {}

    for comp, comp_text in input.items():
        comp_translation = {}

        for k, v in comp_text.items():
            comp_translation[k] = (
                v
                if v != ""
                else ts.translate_text(k, to_language=LANG, translator=TRANSLATOR)
            )

        output[comp] = comp_translation

    return output


if __name__ == "__main__":
    logging.info('1. Update secretflow comp list.')
    with open(COMP_LIST_FILE, 'w') as f:
        json.dump(json.loads(MessageToJson(COMP_LIST)), f, indent=2, ensure_ascii=False)

    logging.info('2. Update translation.')

    with open(TRANSLATION_FILE, "r") as f:
        archieve = json.load(f)

    trans = translate(gettext(COMP_LIST, archieve))

    with open(TRANSLATION_FILE, "w") as f:
        json.dump(trans, f, indent=2, ensure_ascii=False)
