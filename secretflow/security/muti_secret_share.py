import random
def egcd(a, b):
    """
    Extended euclidean algorithm to find integer(x,y) that satisfy ax+by = gcd(a,b)
    """
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)
def mod_inverse(k, prime):
    """
    Returns a value r that kr ≡ 1(mod prime)
    """
    k = k % prime
    if k < 0:
        r = egcd(prime, -k)[2]
    else:
        r = egcd(prime, k)[2]
    return (prime + r) % prime

def random_polynomial(degree,upper_bound):
    """
    Generates a random polynomial with positive coefficients.
    """
    if degree < 0:
        raise ValueError('Degree must be a non-negative number.')
    coefficients = []
    r = random.SystemRandom()
    for i in range(degree):
        random_coeff = r.randint(0,upper_bound - 1)
        coefficients.append(random_coeff)
    return coefficients
def get_polynomial_points(coefficients,secret_int_list, num_points, prime,seed):
    """
    Calculates the first n polynomial points.

    Args:
        coefficients:        Coefficients of random degree t polynomial q(x)
        secret_int_list:     Secret list to be shared
        num_points:          The number of players
        prime:               The degree of F
    Returns:
        points:             [ (1, p(1)), (2, p(2)), ... (n, p(n)) ]
    """
    points = []
    e_vector = []
    # Pre-selected elements of F with length len(secret_int_list) that are known to the Dealer and all n players.
    random.seed(seed)
    for i in range(len(secret_int_list)):
        rng = random.randint(0,prime - 1)
        e_vector.append(rng)

    for x in range(1,num_points+1):
        # start with x=1 and calculate the value of y
        y = 0
        for i in range(len(coefficients)):
            # evaluate the q(x)
            exponentiation = (x ** i) % prime
            term = (coefficients[i] * exponentiation) % prime
            y = (y + term) % prime
        for i in range(len(secret_int_list)):
            # evaluate the (x - e_1)*...*(x- e_k)
            y = (y * (x - e_vector[i])) % prime
        second_section = 0
        for i in range(len(secret_int_list)):
            # evaluate the lagrange basis polynomial l_i(x)
            numerator, denominator = 1, 1
            for j in range(len(e_vector)):
                # don't compute a polynomial fraction if i equals j
                if i == j:
                    continue
                # compute a fraction & update the existing numerator + denominator
                numerator = (numerator * (x - e_vector[j])) % prime
                denominator = (denominator * (e_vector[i] - e_vector[j])) % prime
            # get the polynomial from the numerator * denominator mod inverse
            lagrange_polynomial = (numerator * mod_inverse(denominator, prime)) % prime
            # get the polynomial from the secret_int_list[i] * lagrange_polynomial
            second_section += (secret_int_list[i] * lagrange_polynomial) % prime
        y = (y + second_section) % prime
        points.append((x, y))
    return points

def modular_lagrange_interpolation(points, secret_length, prime,seed):

    """
    Reconstructing Secrets Using lagrange interpolation

    Args:
        points:          [ (1, p(1)), (2, p(2)), ... (n, p(n)) ]
        secret_length:   length of reconstructed secret

    Return:
        f_x_list：       the secrets of reconstruction
    """
    e_vector = []
    # generate random vector [e_1,...,e_k]
    random.seed(seed)
    for j in range(secret_length):
        ran_int = random.randint(0, prime - 1)
        e_vector.append(ran_int)
    # break the points up into lists of x and y values
    x_values, y_values = zip(*points)
    # initialize f(x) and begin the calculation: f(x) = SUM( y_i * l_i(x) )
    f_x_list = []
    # save all reconstruction secrets s_i,...,s_k
    for m in range(secret_length):
        f_x = 0
        for i in range(len(points)):
            # evaluate the lagrange basis polynomial l_i(x)
            numerator, denominator = 1, 1
            for j in range(len(points)):

                # don't compute a polynomial fraction if i equals j
                if i == j:
                    continue

                # compute a fraction & update the existing numerator + denominator
                numerator = (numerator * (e_vector[m] - x_values[j])) % prime
                denominator = (denominator * (x_values[i] - x_values[j])) % prime

            # get the polynomial from the numerator + denominator mod inverse
            lagrange_polynomial = numerator * mod_inverse(denominator, prime)

            # multiply the current y & the evaluated polynomial & add it to f(x)
            f_x = ( f_x + (int(y_values[i]) * lagrange_polynomial)) % prime

        f_x_list.append(f_x)
    return f_x_list
class MutiSecretSharer():
    def __init__(self):
        pass
    @staticmethod
    def split_secret(secret_int_list,num_points, point_threshold, prime,seed):
        """ Split a secret (integer) into shares (pair of integers / x,y coords).

               Sample the points of a random polynomial with the y intercept equal to
               the secret int.
           """
        if point_threshold < 2:
            raise ValueError("Threshold must be >= 2.")
        if point_threshold > num_points:
            raise ValueError("Threshold must be < the total number of points.")
        if not prime:
            raise ValueError("Error!You need to specify a finite field")
        coefficients = random_polynomial(point_threshold - len(secret_int_list), prime)
        points = get_polynomial_points(coefficients, secret_int_list, num_points, prime,seed)
        return points

    @staticmethod
    def recover_secret(points, secret_length, prime,seed):
        """ Join int points into a secret int.

                        Get the intercept of a random polynomial defined by the given points.
                    """
        if not isinstance(points, list):
            raise ValueError("Points must be in list form.")
        for point in points:
            if not isinstance(point, tuple) and len(point) == 2:
                raise ValueError("Each point must be a tuple of two values.")
        if not prime:
            raise ValueError("Error!You need to specify a finite field")
        secret_int_list = modular_lagrange_interpolation(points , secret_length, prime,seed)
        return secret_int_list