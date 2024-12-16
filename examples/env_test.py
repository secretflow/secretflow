from fed import init
import secretflow as sf

sf.init(parties=['alice', 'bob', 'carol'], address='local')
alice_device = sf.PYU('alice')
message_from_alice = alice_device(lambda x: x)("Hello World!")
print(message_from_alice)
print(sf.reveal(message_from_alice))
