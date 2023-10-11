from he_shamir_agg import *

sf.init(['alice', 'bob', 'carol', 'davy'], address='local')

alice, bob, carol, davy = (
    sf.PYU('alice'),
    sf.PYU('bob'),
    sf.PYU('carol'),
    sf.PYU('davy'),
)


threshold = 3
trusted_auth = TA(
    device=alice, participants=[alice, bob, carol, davy], threshold=threshold
)


aggregator = SecureAggregator(alice, trusted_auth, [alice, bob, carol, davy])

print('parties: alice, bob, carol, davy')

print('threshold:' + str(threshold))

a = alice(lambda: np.random.rand(2, 5))()
b = bob(lambda: np.random.rand(2, 5))()
c = carol(lambda: np.random.rand(2, 5))()
d = davy(lambda: np.random.rand(2, 5))()


print('data[alice]:')
print(sf.reveal(a))
print('data[bob]:')
print(sf.reveal(b))
print('data[carol]:')
print(sf.reveal(c))
print('data[davy]:')
print(sf.reveal(d))

print()
print('(Case1) active parties: alice, davy')
sum_a_d = aggregator.sum([a, d], axis=0)
print('data[aggregation]:')
print(sf.reveal(sum_a_d))
print('aggregation failed!')

print()
print('(Case 2) active parties: alice, bob, davy')
sum_a_b_d = aggregator.sum([a, b, d], axis=0)
print('data[aggregation]:')
print(sf.reveal(sum_a_b_d))
print('aggregation successed!')
