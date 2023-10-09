import configparser
import math
import random

import sympy
from numba import jit
from phe import paillier


class ThresholdPaillier(object):
    def __init__(
        self,
        size_of_n,
        num_clients,
        threshold,
        load=False,
        store=False,
        param_file=None,
    ):
        # the number of parties
        self.l = num_clients
        # the threshold
        self.t = threshold

        if load == True:
            config = configparser.ConfigParser()
            config.read(param_file, encoding="utf-8")
            self.n = int(config["pp"]["n"])
            self.ns = int(config["pp"]["ns"])
            self.nSPlusOne = int(config["pp"]["nSPlusOne"])
            self.delta = int(config["pp"]["delta"])
            self.combineSharesConstant = int(config["pp"]["combineSharesConstant"])
            self.viarray = [0] * self.l
            self.si = [0] * self.l
            for i in range(self.l):
                self.viarray[i] = int(config["public"]["v" + str(i)])
                self.si[i] = int(config["private"]["s" + str(i)])
            self.r = int(config["private"]["r"])
            self.v = int(config["public"]["v"])

        else:
            pub, priv = paillier.generate_paillier_keypair(n_length=size_of_n)
            self.p1 = priv.p
            self.q1 = priv.q

            # search for strong primes
            while sympy.isprime(2 * self.p1 + 1) != True:
                pub, priv = paillier.generate_paillier_keypair(n_length=size_of_n)
                self.p1 = priv.p
            while sympy.isprime(2 * self.q1 + 1) != True:
                pub, priv = paillier.generate_paillier_keypair(n_length=size_of_n)
                self.q1 = priv.q
            # strong primes
            self.p = (2 * self.p1) + 1
            self.q = (2 * self.q1) + 1
            # N, the modulus
            self.n = self.p * self.q
            # multiple factor
            self.s = 1
            # N^s = N
            self.ns = pow(self.n, self.s)
            # N^{s+1}, i.e., N^2
            self.nSPlusOne = pow(self.n, self.s + 1)
            # N+1
            self.nPlusOne = self.n + 1
            # N^2
            self.nSquare = self.n * self.n
            # M
            self.m = self.p1 * self.q1
            # NM
            self.nm = self.n * self.m
            # \varDelta = l!
            self.delta = self.factorial(self.l)
            self.rnd = random.randint(1, 1e50)
            # \frac{1}{4\varDelta^2} mod N
            self.combineSharesConstant = sympy.mod_inverse(
                (4 * self.delta * self.delta) % self.n, self.n
            )
            # decryption private key SK, here choose \beta as M^{-1} mod N
            self.d = self.m * sympy.mod_inverse(self.m, self.n)
            # choose random coefficients, a_0 = self.d = SK
            self.ais = [self.d]
            # loop for t-1 times
            for i in range(1, self.t):
                self.ais.append(random.randint(0, self.nm - 1))
            # below are parameters for validity checking
            self.r = random.randint(
                1, self.p
            )  # Need to change upper limit from p to one in paper
            while math.gcd(self.r, self.n) != 1:
                self.r = random.randint(0, self.p)
            # VK
            self.v = (self.r * self.r) % self.nSquare

            self.si = [0] * self.l
            self.viarray = [0] * self.l
            # compute private key shares and verification key shares
            for i in range(self.l):
                self.si[i] = 0
                X = i + 1
                for j in range(self.t):
                    self.si[i] += self.ais[j] * pow(X, j)
                self.si[i] = self.si[i] % self.nm
                self.viarray[i] = pow(self.v, self.si[i] * self.delta, self.nSquare)

        self.priv_keys = []
        for i in range(self.l):
            self.priv_keys.append(
                ThresholdPaillierPrivateKey(
                    self.n,
                    self.l,
                    self.combineSharesConstant,
                    self.t,
                    self.v,
                    self.viarray,
                    self.si[i],
                    i + 1,
                    self.r,
                    self.delta,
                    self.nSPlusOne,
                )
            )
        self.pub_key = ThresholdPaillierPublicKey(
            self.n,
            self.nSPlusOne,
            self.r,
            self.ns,
            self.t,
            self.delta,
            self.combineSharesConstant,
        )
        if store == True:
            self.save_params(param_file)

    def factorial(self, n):
        fact = 1
        for i in range(1, n + 1):
            fact *= i
        return fact

    def compute_GCD(self, x, y):
        while y:
            x, y = y, x % y
        return x

    def save_params(self, param_file='params.ini'):
        config = configparser.ConfigParser()
        # clear data
        file = open(param_file, 'w').close()
        config.read(param_file, encoding="utf-8")
        if not config.has_section("pp"):
            config.add_section("pp")
        config.set("pp", "n", str(self.n))
        config.set("pp", "ns", str(self.ns))
        config.set("pp", "nSPlusOne", str(self.nSPlusOne))
        config.set("pp", "delta", str(self.delta))
        config.set("pp", "combineSharesConstant", str(self.combineSharesConstant))

        if not config.has_section("private"):
            config.add_section("private")
        for i in range(self.l):
            config.set("private", "s" + str(i), str(self.si[i]))
        config.set("private", "r", str(self.r))

        if not config.has_section("public"):
            config.add_section("public")
        config.set("public", "v", str(self.v))
        for i in range(self.l):
            config.set("public", "v" + str(i), str(self.viarray[i]))

        config.write(open(param_file, 'w'))


class PartialShare(object):
    def __init__(self, share, server_id):
        self.share = share
        self.server_id = server_id


class ThresholdPaillierPrivateKey(object):
    def __init__(
        self,
        n,
        l,
        combineSharesConstant,
        t,
        v,
        viarray,
        si,
        server_id,
        r,
        delta,
        nSPlusOne,
    ):
        self.n = n
        self.l = l
        self.combineSharesConstant = combineSharesConstant
        self.t = t
        self.v = v
        self.viarray = viarray
        self.si = si
        self.server_id = server_id
        self.r = r
        self.delta = delta
        self.nSPlusOne = nSPlusOne

    def partial_decrypt(self, c):
        return PartialShare(
            pow(c.c, self.si * 2 * self.delta, self.nSPlusOne), self.server_id
        )


class ThresholdPaillierPublicKey(object):
    def __init__(self, n, nSPlusOne, r, ns, t, delta, combineSharesConstant):
        self.n = n
        self.nSPlusOne = nSPlusOne
        self.r = r
        self.ns = ns
        self.t = t
        self.delta = delta
        self.combineSharesConstant = combineSharesConstant

    def encrypt(self, msg):
        msg = msg % self.nSPlusOne if msg < 0 else msg
        c = (
            pow(self.n + 1, msg, self.nSPlusOne) * pow(self.r, self.ns, self.nSPlusOne)
        ) % self.nSPlusOne
        return EncryptedNumber(c, self.nSPlusOne, self.n)


class EncryptedNumber(object):
    def __init__(self, c, nSPlusOne, n):
        self.c = c
        self.nSPlusOne = nSPlusOne
        self.n = n

    def __mul__(self, cons):
        if cons < 0:
            return EncryptedNumber(
                pow(sympy.mod_inverse(self.c, self.nSPlusOne), -cons, self.nSPlusOne),
                self.nSPlusOne,
                self.n,
            )
        else:
            return EncryptedNumber(
                pow(self.c, cons, self.nSPlusOne), self.nSPlusOne, self.n
            )

    def __add__(self, c2):
        return EncryptedNumber((self.c * c2.c) % self.nSPlusOne, self.nSPlusOne, self.n)


def combine_shares(shrs, w, delta, combineSharesConstant, nSPlusOne, n, ns):
    cprime = 1
    for i in range(w):
        ld = delta
        for iprime in range(w):
            if i != iprime:
                if shrs[i].server_id != shrs[iprime].server_id:
                    ld = (ld * -shrs[iprime].server_id) // (
                        shrs[i].server_id - shrs[iprime].server_id
                    )
        # print(ld)
        shr = sympy.mod_inverse(shrs[i].share, nSPlusOne) if ld < 0 else shrs[i].share
        ld = -1 * ld if ld < 1 else ld
        temp = pow(shr, 2 * ld, nSPlusOne)
        cprime = (cprime * temp) % nSPlusOne
    L = (cprime - 1) // n
    result = (L * combineSharesConstant) % n
    return result - ns if result > (ns // 2) else result
