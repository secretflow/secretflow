# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_separator(color=bcolors.OKBLUE):
    """ Prints a highlighted message separated by dashes
    """
    try:
        max_length = os.get_terminal_size().columns
    except:
        max_length = 80
    print(f"{color}{''.join(['-'] * max_length)}{bcolors.ENDC}")


def print_response(content, color=bcolors.OKGREEN):
    """ Prints a highlighted message separated by dashes
    -------
    content
    -------
    """
    if type(content) == str:
        content = [content]
    for content_i in content:
        print(f"{color} {content_i} {bcolors.ENDC}")

def print_highlighted(content, color=bcolors.OKGREEN):
    """ Prints a highlighted message separated by dashes
    -------
    content
    -------
    """
    if type(content) == str:
        content = [content]
    try:
        max_length = os.get_terminal_size().columns
    except:
        max_length = 80
    print(f"\n{color}{''.join(['-'] * max_length)}")
    for content_i in content:
        print(content_i)
    print(f"{''.join(['-'] * max_length)}{bcolors.ENDC}")
    print()

def print_warning(content):
    return print_highlighted(content, color=bcolors.WARNING)

def print_dict_highlighted(content: dict, title=None, color=bcolors.OKGREEN):
    if len(content.keys()) == 0:
        print()
        return
    lfill = max(map(lambda key: len(str(key)), list(content.keys())))
    try:
        max_length = os.get_terminal_size().columns
    except:
        max_length = 80
    print(f"\n{color}{''.join(['-'] * max_length)}")
    if title is not None:
        print(f"Title: {title}")
    for key, value in sorted(content.items()):
        key = str(key)
        if isinstance(value, float):
            print(f"{key.ljust(lfill)}: {value:.4f}")
        elif isinstance(value, list):
            print(f"{key.ljust(lfill)} (len={len(value)}): {value}")
        else:
            print(f"{key.ljust(lfill)}: {value}")
    print(f"{''.join(['-'] * max_length)}{bcolors.ENDC}")
    print()