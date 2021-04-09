import pandas as pd
from numpy.linalg import inv
from numpy import linspace, meshgrid
from scipy.stats import jarque_bera
import matplotlib.pyplot as plt

# datos sobre presupuesto en canales de publicidad y ventas
# Unidades en miles de dolares
data = pd.read_csv("marketing.csv")

data["bias"] = 1
y = data["sales"]
x = data[["bias", "youtube", "newspaper"]]
x1, x2 = data["youtube"], data["newspaper"]
b = inv( x.T.dot(x) ).dot( x.T ).dot( y )
y_hat = x.dot(b)

min_x1, max_x1 = min(x["youtube"]), max(x["youtube"])
min_x2, max_x2 = min(x["newspaper"]), max(x["newspaper"])

X1, X2 = meshgrid( linspace(min_x1, max_x1, 2), linspace(min_x2, max_x2, 2) )
Y = b[0] + b[1]*X1 + b[2]*X2

fig = plt.figure()
ax = fig.add_subplot(projection ="3d")

surf = ax.plot_surface(X1, X2, Y, alpha=0.3)
ax.scatter(x1, x2, y, alpha=1)

ax.set_xlabel("youtube")
ax.set_ylabel("periodico")
ax.set_zlabel("ventas")

plt.show()

error = y - y_hat
print( jarque_bera(error) )
plt.hist( error, bins=25, edgecolor="k" )
plt.xlabel( "error" )
plt.ylabel( "cantidad" )
plt.show()
