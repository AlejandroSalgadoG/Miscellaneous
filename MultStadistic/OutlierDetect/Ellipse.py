import pandas as pd
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from scipy.stats import chi2

data = pd.read_csv( "pulpfiber_simp.csv" )

x, y = data["X1"], data["X2"]

center = (x.mean(), y.mean())
rad = chi2.ppf( 0.975, 4 )

s = np.cov(x, y)
eig_val, eig_vec = np.linalg.eig(s)

width, height = np.sqrt( np.sort(eig_val) * rad)

max_eig_vec = eig_vec[ :, np.argmax( eig_val ) ]
angle_rad = np.arctan( max_eig_vec[0] / max_eig_vec[1] )
angle_deg = np.rad2deg( angle_rad )

ax = plt.subplot(111)
ellipse = Ellipse(xy=center, width=width*2, height=height*2, angle=angle_deg*-1, alpha=0.1)
ax.add_artist(ellipse)
plt.scatter(x, y)
plt.show()
