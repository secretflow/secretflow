import subprocess
import tempfile


def test_dumps_should_ok():
    code_block_1 = '''
    import numpy as np


    def avg(data):
    return np.average(data, axis=1)

    from secretflow.utils.cloudpickle import code_position_independent_dumps as dumps
    print(dumps(avg))
    '''

    code_block_2 = '''
    import numpy as np


    def foo():
    pass

    def avg(data):
    return np.average(data, axis=1)

    from secretflow.utils.cloudpickle import code_position_independent_dumps as dumps
    print(dumps(avg))
    '''

    with tempfile.TemporaryDirectory() as tmp_dir:
        with open(f'{tmp_dir}/1.py', 'w') as f1:
            f1.write(code_block_1)

        with open(f'{tmp_dir}/2.py', 'w') as f2:
            f2.write(code_block_2)

        p1 = subprocess.run(f'python {tmp_dir}/1.py', capture_output=True, shell=True)
        p2 = subprocess.run(f'python {tmp_dir}/2.py', capture_output=True, shell=True)
        assert p1.stdout == p2.stdout
