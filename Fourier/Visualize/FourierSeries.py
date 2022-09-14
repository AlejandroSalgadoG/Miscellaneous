import numpy as np
import matplotlib.pyplot as plt


class Bezier:
    def __init__(self, p1, d1, d2, d3):
        x, y = p1
        self.p1 = p1
        self.p2 = (x + d1[0], y + d1[1])
        self.p3 = (x + d2[0], y + d2[1])
        self.p4 = (x + d3[0], y + d3[1])

        self.inv_p1 = self.invert_y(self.p1)
        self.inv_p2 = self.invert_y(self.p2)
        self.inv_p3 = self.invert_y(self.p3)
        self.inv_p4 = self.invert_y(self.p4)

    def interpol(self, p1, p2, t):
        x1, y1 = p1
        x2, y2 = p2
        return (1-t)*x1 + t*x2, (1-t)*y1  + t*y2

    def invert_y(self, p):
        x, y = p
        return x, (250 - y)

    def calculate(self, t):
        l1 = self.interpol(self.inv_p1, self.inv_p2, t)
        l2 = self.interpol(self.inv_p2, self.inv_p3, t)
        l3 = self.interpol(self.inv_p3, self.inv_p4, t)

        q1 = self.interpol(l1, l2, t)
        q2 = self.interpol(l2, l3, t)

        return self.interpol(q1, q2, t)


class Function:
    def __init__(self):
        self.n_curves = 0
        self.curves = []

    def add_curve(self, bezier_curve):
        self.curves.append(bezier_curve)
        self.n_curves += 1

    def eval(self, x):
        if x == 1: return self.curves[-1].calculate(1)
        new_x = x * self.n_curves
        pos = int(new_x)
        local_x = new_x - pos
        return self.curves[min(pos, self.n_curves-1)].calculate(local_x)

    def get_points(self, n):
        for x in np.linspace(0, 1, n):
            yield x, self.eval(x)


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

function = Function()
function.add_curve(bezier1)
function.add_curve(bezier2)
function.add_curve(bezier3)
function.add_curve(bezier4)
function.add_curve(bezier5)
function.add_curve(bezier6)
function.add_curve(bezier7)
function.add_curve(bezier8)
function.add_curve(bezier9)
function.add_curve(bezier10)

def comp_mult(p1, p2):
    p1_real, p1_imag = p1
    p2_real, p2_imag = p2
    return (p1_real*p2_real - p1_imag*p2_imag), (p1_real*p2_imag + p1_imag*p2_real)

def comp_e(t):
    return np.cos(t), np.sin(t)

def calc_coef(n, dt):
    real, imag = 0, 0
    for t in np.arange(0, 1, dt):
        f = function.eval(t)
        e = comp_e(-n * 2*np.pi* t)
        r_real, r_imag = comp_mult(f, e)
        real += dt * r_real
        imag += dt * r_imag
    return real, imag

def eval_serie(t, N):
    f_real, f_imag = 0, 0
    for n in range(-N, N+1):
        e = comp_e(n * 2*np.pi * t)
        c = calc_coef(n, dt=0.005)
        r_real, r_imag = comp_mult(c, e)
        f_real += r_real
        f_imag += r_imag
    return f_real, f_imag

f_real, f_imag = [], []
f_hat_real, f_hat_imag = [], []

for t in np.linspace(0,1,150):
    f = function.eval(t)
    f_real.append(f[0])
    f_imag.append(f[1])

    f_hat = eval_serie(t, N=100)
    f_hat_real.append(f_hat[0])
    f_hat_imag.append(f_hat[1])

    print(t, f, f_hat)

fig, axis = plt.subplot_mosaic([
    ["real", "complex"],
    ["imag", "complex"],
])

axis["real"].plot(f_real)
axis["real"].plot(f_hat_real)

axis["imag"].plot(f_imag)
axis["imag"].plot(f_hat_imag)

axis["complex"].plot(f_real, f_imag)
axis["complex"].plot(f_hat_real, f_hat_imag)

plt.show()

print("done")