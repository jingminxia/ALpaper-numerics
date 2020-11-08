"""
Plot the L2/H1error-h figure for gamma=10^4, 10^5, 10^6 using the ALMG-PBJ solver
"""

import numpy as np
import matplotlib.pyplot as plt

h = np.array([2**(-1), 2**(-2), 2**(-3), 2**(-4)]) * 0.1
h2 = h**2

fig1 = plt.figure(1)
plt.title(r'$\gamma=10^4$')
plt.xlabel("h")
plt.ylabel("error")
l2error1 = [2.585534e-09, 3.505911e-10,2.666140e-11,1.908460e-12]
h1error1 = [1.111607e-07,1.004500e-07,1.630215e-08,2.289706e-09]
# adjust the location of the reference lines
plt.loglog(h, h2/10**3, label=r'$h^2$', linestyle='dotted', color='gray', linewidth=3)
plt.loglog(h, h**3/10**4, label=r'$h^3$', linestyle='-.', color='gray', linewidth=3)
plt.loglog(h, l2error1, label=r'$||\mathbf{n}-\mathbf{n}_h||_0$', marker='s', color='blue', markerfacecolor='none', markeredgecolor='blue', markersize=18, markeredgewidth=3, linewidth=3)
plt.loglog(h, h1error1, label=r'$||\mathbf{n}-\mathbf{n}_h||_1$', marker='o', color='red', markerfacecolor='none', markeredgecolor='red', markersize=18, markeredgewidth=3, linewidth=3)
fig1.savefig("gamma-e4.pdf")


fig2 = plt.figure(2)
plt.title(r'$\gamma=10^5$')
plt.xlabel("h")
plt.ylabel("error")
l2error2 = [2.617826e-09,3.497285e-10,2.654390e-11,1.805547e-12]
h1error2 = [1.162894e-07,9.995307e-08,1.621614e-08,2.248782e-09]
plt.loglog(h, h2/(10**3), label=r'$h^2$', linestyle='dotted', color='gray', linewidth=3)
plt.loglog(h, h**3/(10**4), label=r'$h^1$', linestyle='-.', color='gray', linewidth=3)
plt.loglog(h, l2error2, label=r'$||\mathbf{n}-\mathbf{n}_h||_0$', marker='s', color='blue', markerfacecolor='none', markeredgecolor='blue', markersize=18, markeredgewidth=3, linewidth=3)
plt.loglog(h, h1error2, label=r'$||\mathbf{n}-\mathbf{n}_h||_1$', marker='o', color='red', markerfacecolor='none', markeredgecolor='red', markersize=18, markeredgewidth=3, linewidth=3)
fig2.savefig("gamma-e5.pdf")

fig4 = plt.figure(3)
plt.title(r'$\gamma=10^6$')
plt.xlabel("h")
plt.ylabel("error")
l2error4 = [3.759225e-09,3.503299e-10,2.656177e-11,1.798210e-12]
h1error4 = [2.103225e-07,1.000306e-07,1.621217e-08,2.239675e-09]
plt.loglog(h, h2/(10**3), label=r'$h^2$', linestyle='dotted', color='gray', linewidth=3)
plt.loglog(h, h**3/(10**4), label=r'$h^3$', linestyle='-.', color='gray', linewidth=3)
plt.loglog(h, l2error4, label=r'$||\mathbf{n}-\mathbf{n}_h||_0$', marker='s', color='blue', markerfacecolor='none', markeredgecolor='blue', markersize=18, markeredgewidth=3, linewidth=3)
plt.loglog(h, h1error4, label=r'$||\mathbf{n}-\mathbf{n}_h||_1$', marker='o', color='red', markerfacecolor='none', markeredgecolor='red', markersize=18, markeredgewidth=3, linewidth=3)
fig4.savefig("gamma-e6.pdf")


# plot the legend separately
figlegend = plt.figure()
ax_leg = figlegend.add_subplot(111)
labels = [r"$h^2$", r"$h^3$", r'$||\mathbf{n}-\mathbf{n}_h||_0$', r'$||\mathbf{n}-\mathbf{n}_h||_1$']
f1 = lambda l: plt.plot([], [], linestyle=l, color='gray', linewidth=2)[0]
f2 = lambda m,c: plt.plot([], [], marker=m, color=c, markerfacecolor='none', markeredgecolor=c, markersize=14, linewidth=2)[0]
handles = [f1('dotted'), f1('-.'), f2('s', 'blue'), f2('o', 'red')]
ax_leg.legend(handles, labels, loc='center')
ax_leg.axis('off')
figlegend.savefig("legendbox.pdf")
