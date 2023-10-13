import logging


def config_logger(mode="log", fname="output.log"):
    logger = logging.getLogger(mode)
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    fh = logging.FileHandler(fname, "a", encoding="utf-8")

    # formatter = logging.Formatter('%(asctime)s-%(filename)s-%(message)s')
    formatter = logging.Formatter("%(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)


def print_log(msg_list, oriention="print", mode="log", delimiter=""):
    msg = ""
    logger = logging.getLogger(mode)
    for m in msg_list:
        if msg != "":
            msg = msg + delimiter + str(m)
        else:
            msg = str(m)

    if oriention == "print":
        print(msg)
    elif oriention == "logger":
        logger.info(msg)


# config_logger(mode='log', fname='./logs/split.log')
