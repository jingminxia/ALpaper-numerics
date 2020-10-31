import numpy as np
import matplotlib.pyplot as plt

x = [10, 50, 1e2, 5e2, 1e3, 5e3, 1e4, 2e4]
y = [6.865e-9, 2.981e-9, 1.732e-9, 1.467e-9, 1.231e-9, 3.458e-10, 2.221e-10, 8.231e-11]

z = 1e-3*np.power(x, -0.5) 
plt.xlabel(r"$\gamma$")
plt.ylabel(r"$||\mathbf{n}\cdot\mathbf{n}-1||_0$")
plt.loglog(x,y, "b", marker="s", markersize=10, linewidth=1.0)
plt.loglog(x,z, "k", marker="", linestyle='dashed')
#plt.xlim(1e4, 1e9)
#plt.legend()
plt.autoscale(enable=True,axis='both',tight=None)
plt.savefig('./error.pdf')

