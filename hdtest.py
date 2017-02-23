#%%
import numpy
import scipy
import os
os.chdir('/home/bbales2/diffusion')
import pyximport
import time
pyximport.install(reload_support = True)

import matplotlib.pyplot as plt

data = """0	2.496643737
5.6568	2.760147955
12.0599	2.731482238
18.4631	2.59114926
24.1199	2.48375488
29.7768	2.432533019
36.1799	2.240920246
41.8368	2.232590513
48.2399	2.161828643
53.8967	2.059430461
60.2999	1.996965007
65.9567	1.887728518
72.3598	1.907844208
78.017	1.808829964
84.42	1.57142775
90.077	1.701358373
95.734	1.605451805
102.137	1.456515395
108.54	1.461104922
114.197	1.352972172
119.854	1.302182066
126.257	1.243359915
132.66	1.066671415
138.317	1.159614932
143.973	1.126754173
151.045	1.070932217
156.701	1.032455722
162.358	1.045294806
168.015	0.96000336
174.418	0.893994653
180.821	0.903897486
186.478	0.830675659
192.135	0.774701278
198.538	0.718758126
204.941	0.716032941
210.598	0.713599868
216.255	0.630989886
222.658	0.648870892
228.315	0.567879499
234.718	0.541446439
240.375	0.548667775
246.778	0.468353856
252.435	0.495793694
258.838	0.450233883
264.495	0.435563393
270.898	0.403205311
276.555	0.427295594
282.212	0.359591388
289.283	-0.582969908
294.94	0.335402222
300.597	0.309952966
306.254	0.300129552
313.325	0.293129426
318.981	0.293583798
324.638	0.239513564
330.295	0.255303141
336.698	0.220204387
343.101	0.225505408
348.758	0.227246057
354.415	0.154325388
360.818	0.199752451
367.221	0.169280343
372.878	0.176321289
378.535	0.167477769
384.938	0.145899825
390.595	0.125536605
396.998	0.12135882
402.655	0.114396878
409.058	0.11778856
414.715	0.118246595
421.118	0.102681041
426.775	0.085488536
433.178	0.070239495
438.835	0.093348714
445.238	0.088575438
451.641	0.077767637
457.298	0.078043344
462.955	0.071393113
468.612	0.041080937
475.683	0.058835859
481.34	0.048843231
486.997	0.051745413
492.654	0.06697217"""

dataList = []
for line in data.split('\n')[0::2]:
    d, at = line.split()

    d = float(d)
    at = float(at)

    if at > 0.0:
        dataList.append((d, at))

data = numpy.array(dataList)

xs = numpy.array(data[1:, 0])
at = numpy.array(data[1:, 1])

nxs = numpy.linspace(0.0, max(xs), len(xs))
at = numpy.interp(nxs, xs, at)
xs = nxs

maxxs = max(xs)
maxat = max(at)

xs /= max(xs)
at /= max(at)
#%%

N = len(xs)
#xs = numpy.linspace(0, 1, N + 2)
dx = xs[1] - xs[0]

D = None
a = 1.0

b1 = 1.0
b2 = 0.0

#for a in [-0.0001, 0.0, 0.0001]:
def f(y, t0, D, a):
    dcdt = numpy.zeros(N)

    co = y

    f = (D * (co[1] - 2 * co[0] + b1) / dx**2 + D * (a / 2.0) * (co[1]**2 - 2 * co[0]**2 + b1**2) / dx**2)

    dcdt[0] = f

    for i in range(1, N - 1):
        f = (D * (co[i + 1] - 2 * co[i] + co[i - 1]) / dx**2 + D * (a / 2.0) * (co[i + 1]**2 - 2 * co[i]**2 + co[i - 1]**2) / dx**2)

        dcdt[i] = f

    f = (D * (b2 - 2 * co[N - 1] + co[N - 2]) / dx**2 + D * (a / 2.0) * (b2**2 - 2 * co[N - 1]**2 + co[N - 2]**2) / dx**2)

    dcdt[N - 1] = f

    return dcdt

def solve(D, a):
    return scipy.integrate.odeint(f, numpy.zeros(N), [0.0, 0.1], args = (D, a), mxstep = 1000)[1]

plt.plot(solve(1.0, 0.2))
plt.show()

#%%
zs = numpy.linspace(0, 5.0, 100)

Dalpha = 0.4
Dbeta = 1.4

aalpha = -0.5
abeta = 1.0

D = lambda x : (Dbeta - Dalpha) * (x + 1.0) / 2.0 + Dalpha
a = lambda y : (abeta - aalpha) * (y + 1.0) / 2.0 + aalpha

zD = lambda D : 2.0 * (D - Dalpha) / (Dbeta - Dalpha) - 1.0
za = lambda a : 2.0 * (a - aalpha) / (abeta - aalpha) - 1.0

#%%

os.chdir('/home/bbales2/gpc')

import gpc

reload(gpc)

minD = 0.4
maxD = 1.4
mina = -0.5
maxa = 1.0

hd = gpc.GPC(7, solve, [('u', (minD, maxD), 7),
                        ('u', (mina, maxa), 7)])
#%%
mn = 0.2

def L(z1, z2):
    logp = 0.0

    u = hd.approx(z1, z2)
    for i in range(1, len(u)):
        logp += -0.5 * (at[i] - u[i])**2 / mn**2 - numpy.log(mn) - 0.5 * numpy.log(2.0) - 0.5 * numpy.log(numpy.pi)

    return numpy.exp(logp)

denominator = hd.measure(L)

post = numpy.zeros((30, 30))
for i, z1 in enumerate(numpy.linspace(minD, maxD, 30)):
    for j, z2 in enumerate(numpy.linspace(mina, maxa, 30)):
        post[i, j] = L(z1, z2) * hd.prior(z1, z2) / denominator#

plt.imshow(post, interpolation = 'NONE', extent = [mina, maxa, maxD, minD])
plt.colorbar()
plt.xlabel('a')
plt.ylabel('D')
plt.title('Posterior')
plt.gcf().set_size_inches((12, 8))
plt.show()
#%%
plt.plot(hd.approx(0.69, 0.3));
plt.plot(at);#solve(0.89, 0.3));
#%%
def loss(z):
    z1 = z[0]
    z2 = z[1]

    print z1, z2

    return -L(z1, z2) * hd.prior(z1, z2)

res = scipy.optimize.minimize(loss, [0.8, -0.3], bounds = ([minD, maxD - 0.01], [mina, maxa - 0.01]))#
#%%
plt.plot(xs, at, 'bo')
plt.plot(xs, solve(0.89178849, -0.3399514), 'r')
plt.plot(xs, solve(0.88, 0.0), 'g')
plt.legend(['data', 'opt', 'maxp'])
plt.show()
#%%
os.chdir('/home/bbales2/gpc')

import gpc

reload(gpc)

minD = 0.5
maxD = 1.2
mina = -0.5
maxa = 0.5

solve2 = lambda a : solve(1.0, a)

hd = gpc.GPC(3, solve2, [('n', (-0.2, 0.2), 3)])

plt.plot(hd.approx(-0.2), 'r')
plt.plot(hd.sols[(0.0,)], 'g')
plt.plot(solve2(-0.2), 'b')
plt.show()

#%%

labels = []
for z1 in numpy.linspace(minD, maxD, 5):
    z2 = mina
    plt.plot(hd.approx(z1, z2), 'r')
    plt.plot(solve(z1, z2), 'b--')
    labels.append("D = {0}, a = {1}".format(z1, z2))

plt.legend(labels)
plt.show()

labels = []
for z2 in numpy.linspace(mina, maxa, 5):
    z1 = maxD
    plt.plot(hd.approx(z1, z2), 'r')
    plt.plot(solve(z1, z2), 'b--')
    labels.append("D = {0}, a = {1}".format(z1, z2))

plt.legend(labels)
plt.show()
#%%
plt.plot(approx(zD(0.85), za(-0.05)), 'r')
plt.plot(approx(zD(0.89), za(-0.34)), 'g')
plt.plot(at, 'bo')
plt.plot(solve(0.89, -0.34), 'rx')
plt.show()
#%%
reload(gpc)
import scipy.stats
#deg = 10
#xh, wh = scipy.special.p_roots(deg)
for x, wx in zip(xxh, wxh):
    for y, wy in zip(xyh, wyh):
        denominator += wx * wy * L(x, y) / ((Dbeta - Dalpha) * (abeta - aalpha))

        #plt.plot(approx(x))
        #plt.plot(data, '*')
        #plt.title("D = {0}, z = {1}".format(D(x), x))
        #plt.show()
denominator /= (1.0 / ((Dbeta - Dalpha) * (abeta - aalpha)))

zs = numpy.linspace(-1.0, 1.0, 100)
post = numpy.zeros((len(zs), len(zs)))
for i, z1 in enumerate(zs):
    for j, z2 in enumerate(zs):
        post[i, j] = L(z1, z2) * (1.0 / ((Dbeta - Dalpha) * (abeta - aalpha))) / denominator

plt.imshow(post, interpolation = 'NONE', extent = [aalpha, abeta, Dbeta, Dalpha])
plt.colorbar()
plt.xlabel('a')
plt.ylabel('D')
plt.title('Posterior')
plt.gcf().set_size_inches((12, 8))
#plt.plot(D(zs), [(1.0 / ((Dbeta - Dalpha) * (abeta - aalpha)))] * len(zs), 'b')
#plt.legend(['Posterior', 'Prior'])
plt.show()
#%%
total = 0.0
for x, w in zip(xh, wh):
    total += w * L(x) * (1.0 / (beta - alpha)) / denominator
print total / (1.0 / (beta - alpha))
#%%
#%%
plt.plot(solve(mu)[0])
plt.show()