#%%

import matplotlib.pyplot as plt
import numpy
import math

N = 5

ys = []

for n in range(N):
    ys.append(numpy.sqrt(2 * numpy.pi) * math.factorial(n))
#%%
xs, ys = numpy.polynomial.hermite_e.hermegauss(deg)
#%%
N = 10

mu = 0.0
sigma = 1.0

efs = []
for k in range(N):
    efs.append(sigma**k / math.factorial(k))

xs = numpy.linspace(-1.0, 10.0, 100)

ys = []
for x in xs:
    ys.append(numpy.exp(mu + sigma**2 / 2.0) * sum(numpy.polynomial.hermite_e.hermeval([x], efs)))

plt.plot(xs, ys)