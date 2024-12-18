import os
import glob
import json
import logging
from typing import Any, Mapping, Iterable, Union, List, Callable, Optional

from tqdm.auto import tqdm


def resolve_globs(glob_paths: Union[str, Iterable[str]]):
    """Returns filepaths corresponding to input filepath pattern(s)."""
    filepaths = []
    if isinstance(glob_paths, str):
        glob_paths = [glob_paths]

    for path in glob_paths:
        filepaths.extend(glob.glob(path))

    return filepaths


def read_jsonlines(filename: str) -> Iterable[Mapping[str, Any]]:
    """Yields an iterable of Python dicts after reading jsonlines from the input file."""
    file_size = os.path.getsize(filename)
    with open(filename) as fp:
        for line in tqdm(fp.readlines(), desc=f'Reading JSON lines from {filename}', unit='lines'):
            try:
                example = json.loads(line)
                yield example
            except json.JSONDecodeError as ex:
                logging.error(f'Input text: "{line}"')
                logging.error(ex.args)
                raise ex


def hf_read_jsonlines(filename: str, 
                   n: Optional[int]=None, 
                   minimal_questions: Optional[bool]=False,
                   unique_questions: Optional[bool] = False) -> Iterable[Mapping[str, Any]]:
    """Yields an iterable of Python dicts after reading jsonlines from the input file.
       Optionally reads only first n lines from file."""
    file_size = os.path.getsize(filename)
    # O(n) but no memory
    with open(filename) as f:
        num_lines= sum(1 for _ in f)
        if n is None: 
            n = num_lines

    # returning a generator with the scope stmt seemed to be the issue, but I am not 100% sure
    # I also don't know if there's a side effect, but I can't see how the scope wouldn't have
    # remained upen in the first place with the original version...
    # with open(filename) as fp:
    def line_generator():
        unique_qc_ids = set()
        # note, I am p sure that readlines is not lazy, returns a list, thus really only the
        # object conversion is lazy
        for i, line in tqdm(enumerate(open(filename).readlines()[:n]), desc=f'Reading JSON lines from {filename}', unit='lines'):
            try:
                full_example = json.loads(line)

                if unique_questions:
                    qc_id = full_example["object"]["qc_id"]
                    if qc_id in unique_qc_ids:
                        continue
                    else:
                        unique_qc_ids.add(qc_id)

                if not minimal_questions:
                    example = full_example
                else:
                    full_example = full_example
                    q_object = full_example["object"]
                    q_object.pop("question_info")
                    example= {}
                    example["object"] = {
                        "answer":q_object["answer"],
                        "clue_spans":q_object["clue_spans"],
                        "qc_id":q_object["qc_id"],
                        "question_text":q_object["question_text"],
                    }
                yield example

            except json.JSONDecodeError as ex:
                logging.error(f'Input text: "{line}"')
                logging.error(ex.args)
                raise ex
    return line_generator


def load_jsonlines(filename: str) -> List[Mapping[str, Any]]:
    """Returns a list of Python dicts after reading jsonlines from the input file."""
    return list(read_jsonlines(filename))


def write_jsonlines(objs: Iterable[Mapping[str, Any]], filename: str, to_dict: Callable = lambda x: x):
    """Writes a list of Python Mappings as jsonlines at the input file."""
    with open(filename, 'w') as fp:
        for obj in tqdm(objs, desc=f'Writing JSON lines at {filename}'):
            fp.write(json.dumps(to_dict(obj)))
            fp.write('\n')


def read_json(filename: str) -> Mapping[str, Any]:
    """Returns a Python dict representation of JSON object at input file."""
    with open(filename) as fp:
        return json.load(fp)


def write_json(obj: Mapping[str, Any], filename: str, indent:int=None):
    """Writes a Python Mapping at the input file in JSON format."""
    with open(filename, 'w') as fp:
        json.dump(obj, fp, indent=indent)


def print_json(d, indent=4):
    print(json.dumps(d, indent=indent))