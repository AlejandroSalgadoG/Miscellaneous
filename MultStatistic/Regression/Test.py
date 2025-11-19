import pandas as pd
from numpy import sqrt, diag
from numpy.linalg import inv
from scipy.stats import t, f

alpha = 0.10

data = pd.read_csv("marketing.csv")
data["bias"] = 1

y = data["sales"]
x = data[["bias", "youtube", "newspaper"]]

n,p = x.shape
p -= 1 # bias

almost_cov = inv( x.T.dot(x) )
b = almost_cov.dot( x.T ).dot( y )

y_hat = x.dot(b)

r_error = y - y_hat
d_error = y - y.mean()

ss_fit = sum(r_error**2)
ss_mean = sum(d_error**2 )
ss_reg = ss_mean - ss_fit

dof_reg = p
dof_fit = n-(p+1)

ms_reg = ss_reg / dof_reg
ms_error = ss_fit / dof_fit

f_stad = ms_reg / ms_error
break_point = f.ppf( 1-alpha, dof_reg, dof_fit )
print( "f stad =", f_stad, "-", "break point", break_point )

cov = ms_error * almost_cov
se_b = sqrt( diag( cov ) )

t_stad = b / se_b
t_stad = t_stad[1:] # bias
break_point = t.ppf( 1 - alpha/2, dof_fit )
print( "t stad =", t_stad, "-", "break point", break_point )
