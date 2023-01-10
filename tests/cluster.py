SELF_PARTY = None


def set_self_party(party: str):
    global SELF_PARTY
    SELF_PARTY = party


def get_self_party() -> str:
    global SELF_PARTY
    return SELF_PARTY


_parties = {
    'alice': {'address': '127.0.0.1:23041'},
    'bob': {'address': '127.0.0.1:23042'},
    'carol': {'address': '127.0.0.1:23043'},
    'davy': {'address': '127.0.0.1:23044'},
}


def cluster():
    return {
        'parties': _parties,
        'self_party': get_self_party(),
    }
