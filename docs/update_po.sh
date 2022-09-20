#!/bin/bash

make gettext
sphinx-intl update -p _build/gettext -l zh_CN

echo "po files has been updated. Please update po files in locales folder."
