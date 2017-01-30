import numpy
import scipy
import itertools
import math

class GPC(object):
    def __init__(self, K, f, vs):
        self.K = K
        self.f = f

        self.vs = vs

        self.xs = []
        self.ws = []
        self.sols = {}
        self.shape = None
        for v in self.vs:
            if v[0] == 'u':
                xs, ws = scipy.special.p_roots(v[2])
            elif v[0] == 'n':
                xs, ws = scipy.special.he_roots(v[2])

            self.xs.append(xs)
            self.ws.append(ws)

        for xs in itertools.product(*self.xs):
            scaled = []
            for v, x in zip(self.vs, xs):
                if v[0] == 'u':
                    scaled.append((v[1][1] - v[1][0]) * (x + 1.0) / 2.0 + v[1][0])
                elif v[0] == 'n':
                    scaled.append(x * v[1][1] + v[1][0])
            self.sols[xs] = f(*scaled)

            if self.shape is None:
                try:
                    self.shape = self.sols[xs].shape
                except Exception as e:
                    print "WARNING: The function 'f' you pass to GPC *must* return a single numpy.array as a result"
                    print "    It's likely this error was raised because the function is returning something else"

                    raise e

        self.ks = []
        for iis in itertools.product(range(K + 1), repeat = len(self.vs)):
            if sum(iis) <= K:
                self.ks.append(iis)

        self.uks = []
        for ks in self.ks:
            u = numpy.zeros(self.shape)
            for iis in itertools.product(*[range(len(xs)) for xs in self.xs]):
                sol_xs = []
                wt = 1.0
                for i, k, v, xs, ws in zip(iis, ks, self.vs, self.xs, self.ws):
                    sol_xs.append(xs[i])
                    wt *= ws[i]

                    if v[0] == 'u':
                        wt *= scipy.special.eval_legendre(k, xs[i]) / (2.0 / (2.0 * k + 1.0))
                    elif v[0] == 'n':
                        wt *= scipy.special.eval_hermitenorm(k, xs[i]) / (numpy.sqrt(2 * numpy.pi) * math.factorial(k))

                u += wt * self.sols[tuple(sol_xs)]

            self.uks.append(u)

    def normed(self, args):
        normed = []
        for v, a in zip(self.vs, args):
            if v[0] == 'u':
                normed.append(2.0 * (a - v[1][0]) / (v[1][1] - v[1][0]) - 1.0)

                if normed[-1] > 1.0 or normed[-1] < -1.0:
                    raise Exception("Parameter value \"{0}\" out of range for variable {1}".format(a, v))
            elif v[0] == 'n':
                normed.append((a - v[1][0]) / v[1][1])

        return normed

    def approx(self, *args):
        normed = self.normed(args)

        u = numpy.zeros(self.shape)
        for i, ks in enumerate(self.ks):
            s = 1.0
            for v, k, z in zip(self.vs, ks, normed):
                if v[0] == 'u':
                    s *= scipy.special.eval_legendre(k, z)
                elif v[0] == 'n':
                    s *= scipy.special.eval_hermitenorm(k, z)
                else:
                    raise Exception("{0} not valid distribution".format(v))

            u += self.uks[i] * s

        return u

    def mean(self):
        return self.uks[0]

    def covar(self):
        N = numpy.product(self.shape)

        covar = numpy.zeros(2 * self.shape) #self.shape is a tuple -- 2 * self.shape concatenates two together

        for ii in range(N):
            iii = numpy.unravel_index(ii, self.shape)
            for jj in range(N):
                jjj = numpy.unravel_index(jj, self.shape)

                covar_ = 0.0
                for i, ks in enumerate(self.ks):
                    if i == 0:
                        continue

                    s = 1.0
                    for v, k in zip(self.vs, ks):
                        if v[0] == 'u':
                            s *= 2.0 / (2.0 * k + 1.0)
                        elif v[0] == 'n':
                            s *= numpy.sqrt(2 * numpy.pi) * math.factorial(k)
                    covar_ += self.uks[i][iii] * self.uks[i][jjj] * s

                covar[iii + jjj] = covar_

        return covar

    def prior(self, *args):
        p = 1.0

        for v, a in zip(self.vs, args):
            if v[0] == 'u':
                normed = 2.0 * (a - v[1][0]) / (v[1][1] - v[1][0]) - 1.0

                if normed > 1.0 or normed < -1.0:
                    raise Exception("Parameter value \"{0}\" out of range for variable {1}".format(a, v))

                p *= 1.0 / (v[1][1] - v[1][0])
            elif v[0] == 'n':
                normed = (a - v[1][0]) / v[1][1]

                p *= numpy.exp(-0.5 * normed**2) / numpy.sqrt(2 * numpy.pi)

        return p

    def measure(self, g):
        u = None
        for iis in itertools.product(*[range(len(xs)) for xs in self.xs]):
            xts = []
            wt = 1.0
            for i, xs, ws in zip(iis, self.xs, self.ws):
                xts.append(xs[i])
                wt *= ws[i]

            scaled = []
            for v, x in zip(self.vs, xts):
                if v[0] == 'u':
                    scaled.append((v[1][1] - v[1][0]) * (x + 1.0) / 2.0 + v[1][0])
                elif v[0] == 'n':
                    scaled.append(x * v[1][1] + v[1][0])

            if u == None:
                u = wt * g(*scaled)
            else:
                u += wt * g(*scaled)

        return u