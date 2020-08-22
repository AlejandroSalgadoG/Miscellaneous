import numpy as np
import matplotlib.pyplot as plt

def e_i(x):
    return np.cos(x), np.sin(x)

def complex_plot(rel, img):
    ax.scatter(rel, img)
    ax.vlines(0, -1.5, 1.5)
    ax.hlines(0, -1.5, 1.5)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

time_jump = np.pi/2

fig, ax = plt.subplots()
complex_plot(1, 0)

def on_key():
    i = [0]
    def plot(event):
        if event.key == "right": i[0] += 1
        if event.key == "left": i[0] -= 1

        ax.cla()
        x_pos, y_pos = e_i(i[0] * time_jump)
        complex_plot(x_pos, y_pos)
        ax.set_title("%d * %f" % (i[0], time_jump))
        plt.draw()
    return plot

fig.canvas.mpl_connect('key_press_event', on_key())
plt.show()
