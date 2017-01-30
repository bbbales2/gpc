#%%

import numpy
import matplotlib.pyplot as plt
import scipy
import os

os.chdir("/home/bbales2/gpc")

import gpc

Is = 1e-15
n = 1.05
Vt = 0.025
R = 1000

def func(Vs):
    def toMinimize(Vd):
        return Is * numpy.exp(Vd / (n * Vt)) * R + Vd - Vs

    Vd = scipy.optimize.brentq(toMinimize, 0.0, 1.0)

    return numpy.array([Vd])

hd = gpc.GPC(5, func, [('n', (5, 0.2), 5)])

#%%

Vs = numpy.linspace(4.0, 6.0, 100)

plt.plot(Vs, hd.prior(Vs))
plt.title('Source voltage pdf (integrates to 1)')
plt.ylabel('pdf')
plt.xlabel('Source voltage')
plt.show()

Vd = numpy.array([hd.approx(Vs_) for Vs_ in Vs])
plt.plot(Vd, hd.prior(Vs))
plt.title('Voltage across diode pdf (integrates to 1)')
plt.ylabel('pdf')
plt.xlabel('Voltage across diode')
plt.show()

#%%

Vs = numpy.linspace(0.1, 10.0, 100)

plt.plot(Vs, [func(Vs_) for Vs_ in Vs])
plt.plot(Vs, [hd.approx(Vs_) for Vs_ in Vs])
plt.xlabel('Source Voltage')
plt.ylabel('Diode Voltage')
plt.legend(['Exact', 'GPC'])
plt.show()
