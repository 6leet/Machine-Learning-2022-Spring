import random_data_generator as random

def estimate(ground_mean, ground_var):
    print('Data point source function: N(' + str(ground_mean) + ', ' + str(ground_var) + ')')
    print()
    _estimate(ground_mean, ground_var)
def _estimate(ground_mean, ground_var, _mean=0, _var=0, n=0, sum=0, M2=0):
    sum = 0
    n = 0
    mean = 0
    var = 0
    dvar = 1
    while abs(dvar) > 1e-8 or n < 2:
        x = random.gaussian(ground_mean, ground_var)[0]
        print('Add data point: ', x)
        sum += x
        n += 1
        delta1 = x - mean
        mean = sum / n
        delta2 = x - mean
        M2 += delta1 * delta2
        dvar = var - M2 / n
        var = M2 / n
        print('Mean =', mean, 'Variance =', var)

estimate(3, 5)