import numpy as np
import matplotlib.pyplot as plt

x = [10, 50, 1e2,5e2, 3e3, 7e3]
y = [7.51361757188125e-09,2.67281247230815e-09,2.05168040289809e-09,1.41534288487427e-09,3.01583497074702e-10, 7.54092160083207e-11]
z = 1e-2*np.power(x, -0.5)
plt.xlabel(r"$\gamma$")
plt.ylabel(r"$||\mathbf{n}\cdot\mathbf{n}-1||_0$")
plt.loglog(x,y, "b", marker="s", markersize=10, linewidth=1.0)
plt.loglog(x,z, "k", marker="", linestyle='dashed')
#plt.xlim(1e4, 1e9)
#plt.legend()
plt.autoscale(enable=True,axis='both',tight=None)
plt.savefig('./error.pdf')

