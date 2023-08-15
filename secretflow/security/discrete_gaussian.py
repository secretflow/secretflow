import random
from fractions import Fraction
import numpy as np

# sample uniformly from range(m)
def sample_uniform(m, rng):
    assert isinstance(m, int)  # python 3
    # assert isinstance(m,(int,long)) #python 2
    assert m > 0
    return rng.randrange(m)


# sample from a Bernoulli(p) distribution
def sample_bernoulli(p, rng):
    assert isinstance(p, Fraction)
    assert 0 <= p <= 1
    m = sample_uniform(p.denominator, rng)
    if m < p.numerator:
        return 1
    else:
        return 0


# sample from a Bernoulli(exp(-x)) distribution
def sample_bernoulli_exp1(x, rng):
    assert isinstance(x, Fraction)
    assert 0 <= x <= 1
    k = 1
    while True:
        if sample_bernoulli(x / k, rng) == 1:
            k = k + 1
        else:
            break
    return k % 2


# sample from a Bernoulli(exp(-x)) distribution
def sample_bernoulli_exp(x, rng):
    assert isinstance(x, Fraction)
    assert x >= 0
    # Sample floor(x) independent Bernoulli(exp(-1))
    # If all are 1, return Bernoulli(exp(-(x-floor(x))))
    while x > 1:
        if sample_bernoulli_exp1(Fraction(1, 1), rng) == 1:
            x = x - 1
        else:
            return 0
    return sample_bernoulli_exp1(x, rng)


# sample from a geometric(1-exp(-x)) distribution
# assumes x is a rational number >= 0
def sample_geometric_exp_slow(x, rng):
    assert isinstance(x, Fraction)
    assert x >= 0
    k = 0
    while True:
        if sample_bernoulli_exp(x, rng) == 1:
            k = k + 1
        else:
            return k


# sample from a geometric(1-exp(-x)) distribution
# assumes x >= 0 rational
def sample_geometric_exp_fast(x, rng):
    assert isinstance(x, Fraction)
    if x == 0: return 0  # degenerate case
    assert x > 0

    t = x.denominator
    while True:
        u = sample_uniform(t, rng)
        b = sample_bernoulli_exp(Fraction(u, t), rng)
        if b == 1:
            break
    v = sample_geometric_exp_slow(Fraction(1, 1), rng)
    value = v * t + u
    return value // x.numerator


# sample from a discrete Laplace(scale) distribution
# Returns integer x with Pr[x] = exp(-abs(x)/scale)*(exp(1/scale)-1)/(exp(1/scale)+1)
# casts scale to Fraction
# assumes scale>=0
def sample_dlaplace(scale, rng=None):
    if rng is None:
        rng = random.SystemRandom()
    scale = Fraction(scale)
    assert scale >= 0
    while True:
        sign = sample_bernoulli(Fraction(1, 2), rng)
        magnitude = sample_geometric_exp_fast(1 / scale, rng)
        if sign == 1 and magnitude == 0: continue
        return magnitude * (1 - 2 * sign)


# compute floor(sqrt(x)) exactly
# only requires comparisons between x and integer
def floorsqrt(x):
    assert x >= 0
    # a,b integers
    a = 0  # maintain a^2<=x
    b = 1  # maintain b^2>x
    while b * b <= x:
        b = 2 * b  # double to get upper bound
    # now do binary search
    while a + 1 < b:
        c = (a + b) // 2  # c=floor((a+b)/2)
        if c * c <= x:
            a = c
        else:
            b = c
    # check nothing funky happened
    # assert isinstance(a,int) #python 3
    # assert isinstance(a,(int,long)) #python 2
    return a


# sample from a discrete Gaussian distribution N_Z(0,sigma2)
# Returns integer x with Pr[x] = exp(-x^2/(2*sigma2))/normalizing_constant(sigma2)
# mean 0 variance ~= sigma2 for large sigma2
# casts sigma2 to Fraction
# assumes sigma2>=0
def sample_dgauss(sigma2, rng=None):
    if rng is None:
        rng = random.SystemRandom()
    sigma2 = Fraction(sigma2)
    if sigma2 == 0: return 0  # degenerate case
    assert sigma2 > 0
    t = floorsqrt(sigma2) + 1
    while True:
        candidate = sample_dlaplace(t, rng=rng)
        bias = ((abs(candidate) - sigma2 / t) ** 2) / (2 * sigma2)
        if sample_bernoulli_exp(bias, rng) == 1:
            return candidate

def sample_discrete_gaussian(sq_scale:float, prime: int, shape, dtype=np.int64):
   """
   Sample from a discrete Gaussian distribution.
   Args:
       sq_scale: variance
       prime: finite field order
       shape: sampled data dimensions
       dtype: type of data

   Returns:
       random varable following a discrete gaussion distribution.
   """
   target_n = int(np.prod(np.array(shape, dtype=np.int64)))
   result  = np.empty(0, dtype=np.int64)
   while np.size(result) < target_n:
       temp = sample_dgauss(sq_scale) % prime
       result = np.append(result,temp)
   return result[:target_n].reshape(shape).astype(dtype)

