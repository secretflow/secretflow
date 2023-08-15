import math
from sympy import *
def calculate_group_size(clients_num,safety_factor,correctness_factor,dropout,corrupt,threshold_factor):
    def f(k, n, beta, gamma, sigma):
        #f = log(2 * n / k) - 2 * k * ((beta - gamma * n / (n - 1)) ** 2) + sigma * log(2)
        f = log((n / k) * pow(gamma + beta, k / 2) + (2 * n / k) * exp( -2 * pow(beta - gamma * n / (n - 1), 2) * k)) + sigma * log(2)
        return f
    def g(k, n, beta, delta, eta):
        g = log(2 * n / k)- 2 * k * (((1-delta) * n / (n - 1)-beta)**2) + eta * log(2)
        return g
    k = symbols('k')
    n = symbols('n')
    beta = symbols('beta')
    gamma = symbols('gamma')
    sigma = symbols('sigma')
    delta = symbols('delta')
    eta = symbols("eta")
    x0 = 100
    y0 = 100
    x_list = [x0]
    y_list = [y0]
    count = 0
    tolerance = 1e-6
    while count < 100:
        if diff(f(k, n, beta, gamma, sigma), k).subs({k: x0, n: clients_num, beta: threshold_factor, gamma: corrupt, sigma: safety_factor}) == 0 or diff(g(k, n, beta, delta, sigma), k).subs({eta: y0, n: clients_num, beta: threshold_factor, delta: dropout, eta: correctness_factor}) == 0:
            break
        else:
            x0 = x0 - (f(x0, clients_num, threshold_factor, corrupt, safety_factor).evalf()) / (
                diff(f(k, n, beta, gamma, sigma), k).evalf(subs={k: x0, n: clients_num, beta: threshold_factor, gamma: corrupt, sigma: safety_factor}))
            y0 = y0 - (g(y0, clients_num, threshold_factor, dropout, correctness_factor).evalf()) /(
                diff(g(k, n, beta, delta, eta), k).evalf(subs={k: y0, n: clients_num, beta: threshold_factor, delta: dropout, eta: correctness_factor}))

            x_list.append(x0)
            y_list.append(y0)
        if len(x_list) > 1:
            count += 1
            error_x = abs((x_list[-1] - x_list[-2]) / x_list[-1])
            error_y = abs((y_list[-1] - y_list[-2]) / y_list[-1])
            if error_x < tolerance and error_y < tolerance:
                return max(x_list[-1],y_list[-1])
        else:
            pass