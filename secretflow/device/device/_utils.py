import inspect


def check_num_returns(fn):
    # inspect.signature fails on some builtin method (e.g. numpy.random.rand).
    # You can wrap a self define function which calls builtin function inside
    # with return annotation to get multi returns for now.
    if inspect.isbuiltin(fn):
        sig = inspect.signature(lambda *arg, **kwargs: fn(*arg, **kwargs))
    else:
        sig = inspect.signature(fn)

    if sig.return_annotation is None or sig.return_annotation == sig.empty:
        num_returns = 1
    else:
        if (
            hasattr(sig.return_annotation, '_name')
            and sig.return_annotation._name == 'Tuple'
        ):
            num_returns = len(sig.return_annotation.__args__)
        elif isinstance(sig.return_annotation, tuple):
            num_returns = len(sig.return_annotation)
        else:
            num_returns = 1

    return num_returns
