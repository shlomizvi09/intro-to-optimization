from functions import *

f = f()
constrains = [g1(), g2(), g3()]
phi = PenaltyFunction()

x_optimal, lambdas = penalty_aggregate(f, constrains, phi)
# for val in lambdas:
#     print(val)


