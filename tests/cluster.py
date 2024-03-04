SELF_PARTY = None


def set_self_party(party: str):
    global SELF_PARTY
    SELF_PARTY = party


def get_self_party() -> str:
    global SELF_PARTY
    return SELF_PARTY


_parties = {
    'alice': {'address': '127.0.0.1:63841'},
    'bob': {'address': '127.0.0.1:63942'},
    'carol': {'address': '127.0.0.1:63743'},
    'davy': {'address': '127.0.0.1:63644'},
}


def cluster():
    return {
        'parties': _parties,
        'self_party': get_self_party(),
    }
