import numpy as np
import matplotlib.pyplot as plt


class Bezier:
    def __init__(self, p1, d1, d2, d3):
        x, y = p1
        self.p1 = p1
        self.p2 = (x + d1[0], y + d1[1])
        self.p3 = (x + d2[0], y + d2[1])
        self.p4 = (x + d3[0], y + d3[1])

    def interpol(self, p1, p2, t):
        x1, y1 = p1
        x2, y2 = p2
        return (1-t)*x1 + t*x2, (1-t)*y1  + t*y2

    def invert_y(self, p):
        x, y = p
        return x, (250 - y)

    def plot(self):
        p1 = self.invert_y(self.p1)
        p2 = self.invert_y(self.p2)
        p3 = self.invert_y(self.p3)
        p4 = self.invert_y(self.p4)

        for t in np.linspace(0, 1, 50):
            l1 = self.interpol(p1, p2, t)
            l2 = self.interpol(p2, p3, t)
            l3 = self.interpol(p3, p4, t)

            q1 = self.interpol(l1, l2, t)
            q2 = self.interpol(l2, l3, t)

            c1 = self.interpol(q1, q2, t)

            plt.scatter(*c1, s=5, c="k")


bezier1 = Bezier(
    p1=(100.000,  100.000),
    d1=(-79.873,  -70.722),
    d2=( 11.121, -118.450),
    d3=( 22.253,  -63.137),
)

bezier2 = Bezier(
    p1=bezier1.p4,
    d1=( 20.708, -63.445),
    d2=(104.700,  -8.091),
    d3=( 17.596,  62.620),
)

bezier3 = Bezier(
    p1=bezier2.p4,
    d1=(106.520, -57.631),
    d2=(141.050,  54.011),
    d3=( 57.962,  42.954),
)

bezier4 = Bezier(
    p1=bezier3.p4,
    d1=( 64.382,  27.210),
    d2=(-26.826, 106.170),
    d3=(-69.031,  -9.571),
)

bezier5 = Bezier(
    p1=bezier4.p4,
    d1=(-2.588, 16.769),
    d2=(14.981, 52.402),
    d3=(19.335, 56.571),
)

bezier6 = Bezier(
    p1=bezier5.p4,
    d1=(3.116,  3.448),
    d2=(4.951,  7.597),
    d3=(1.442, 11.238),
)

bezier7 = Bezier(
    p1=bezier6.p4,
    d1=( -8.026,  8.216),
    d2=(-27.101,  4.880),
    d3=(-29.613, -0.618),
)

bezier8 = Bezier(
    p1=bezier7.p4,
    d1=(-10.127, -24.307),
    d2=( -8.660, -39.005),
    d3=( -5.175, -68.313),
)

bezier9 = Bezier(
    p1=bezier8.p4,
    d1=( -28.256, 91.015),
    d2=(-139.770, 54.862),
    d3=( -79.698,  1.553),
)

bezier10 = Bezier(
    p1=bezier9.p4,
    d1=(-58.924,    2.796),
    d2=(-41.144, -101.040),
    d3=( 64.929,  -33.297),
)

dpi = 96
fig = plt.figure(figsize=(250/dpi, 250/dpi))

bezier1.plot()
bezier2.plot()
bezier3.plot()
bezier4.plot()
bezier5.plot()
bezier6.plot()
bezier7.plot()
bezier8.plot()
bezier9.plot()
bezier10.plot()

plt.xlim([0, 250])
plt.ylim([0, 250])

plt.xticks([])
plt.yticks([])

plt.show()
