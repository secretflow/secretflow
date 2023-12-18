# Benchmarks

We can quickly benchmark via the component test framework.

Currently contained examples:

- WOE
- LR
- XGB

## Steps

1. Prepare 3 Linux machines for testing purpose. We recomment at least 32 CPUs each. Call them test_001, test_002, test_003 for example.
2. Make sure these 3 Linux machines can ssh login into each other without passwords.
3. use root user
4. install docker image secretflow/secretflow-anolis8
5. create conda environment and install secretflow and component test framework for test_001.
6. create a dir to store your code, e.g. /root/sf_test for all machines
7. create logs dir in this dir, e.g. /root/sf_test/logs for all machines
8. copy files in sf_test in this folder to your folder for test_001
9. prepare the test version you want to test, obtain the whl files
10. copy the  *.whl files to /root/sf_test/*.whl for test_001
11. prepare data csvs, e.g. /root/sf_test/data/train_alice.csv and /root/sf_test/data/test_bob.csv
    Note that the test_001 should have all the test files, other machines should have its own data.
    test_002 should have e.g. /root/sf_test/data/test_bob.csv only.
12. change the following file:
    - /root/sf_test/configs/node.json
    - /root/sf_test/configs/data_*.json
13. Read sf_test/README.md for more details on how to run tests in different modes.
