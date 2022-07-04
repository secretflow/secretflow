SPHINX_APIDOC_OPTIONS=members sphinx-apidoc -f -d 2 -t templates -o ./source ../secretflow/
make clean
env PYTHONPATH=$PYTHONPATH:$PWD/.. make html