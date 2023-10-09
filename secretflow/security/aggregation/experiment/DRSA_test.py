import numpy as np
from secure_aggregator_DRSA import SecureAggregator_DRSA

import secretflow as sf

sf.shutdown()
sf.init(['alice', 'bob'], address='local')
alice, bob = sf.PYU('alice'), sf.PYU('bob')

arr0, arr1 = np.random.rand(10, 20), np.random.rand(10, 20)
print('plain aggregation sum:', np.sum([arr0, arr1], axis=0))
a = alice(lambda: arr0)()
b = bob(lambda: arr1)()


secure_aggr_DRSA = SecureAggregator_DRSA(
    device=alice,
    participants=[alice, bob],
    threshold=1,
    pre_computed=True,
    param_file="params_2.ini",
)

result = secure_aggr_DRSA.sum([a, b], axis=0)
print('secure aggregation sum:', result)
