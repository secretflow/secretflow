import hashlib

_BUF_SIZE = 64 * 1024


def sha256sum(filename: str):
    h = hashlib.sha256()
    global _BUF_SIZE
    bs = bytearray(_BUF_SIZE)
    with open(filename, 'rb') as f:
        while num := f.readinto(bs):  # noqa
            h.update(bs[:num])
    return h.hexdigest()
