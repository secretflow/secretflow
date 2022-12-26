#!/bin/bash

usage() { echo "Usage: $0 [-l <en|zh_CN>]" 1>&2; exit 1; }


while getopts ":l:" o; do
    case "${o}" in
        l)
            l=${OPTARG}
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))

if [ -z "${l}" ]; then
    usage
fi

echo "selected language is: ${l}"

if [[ "$l" != "en" && "$l" != "zh_CN" ]]; then
    usage
fi


# SPHINX_APIDOC_OPTIONS=members,autosummary sphinx-apidoc -f -d 2 -t templates -o ./source ../secretflow/
# make clean
env PYTHONPATH=$PYTHONPATH:$PWD/.. make SPHINXOPTS="-D language='${l}'" html
