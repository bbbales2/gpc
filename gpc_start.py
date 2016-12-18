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
import scipy.stats

N = 5

mu = 0.0
sigma = 1.0

efs = []
for k in range(N):
    efs.append(sigma**k / math.factorial(k))

xs = numpy.linspace(-3.0, 3.0, 100)
ps = scipy.stats.norm.pdf(xs)

ys = []
for x in xs:
    ys.append(numpy.exp(mu + sigma**2 / 2.0) * sum(numpy.polynomial.hermite_e.hermeval([x], efs)))

plt.plot(xs, ys)
plt.show()
plt.plot(ys, ps, 'r--')
plt.plot(xs, ps, 'g')
plt.plot(numpy.exp(mu + sigma * xs), ps, 'b')
plt.show()
