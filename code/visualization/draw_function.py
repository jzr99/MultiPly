import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
x = np.arange(-5, 5, 0.1)
y = []
mpl.style.use('dark_background')
for t in x:
    if t >= 0:
        y_1 = 0.5 * math.exp(-t/0.1)
    else:
        y_1 = 1 - 0.5 * math.exp(t/0.1)
    y.append(y_1)
plt.plot(x, y, color = 'azure', linewidth = 2.0)
plt.xlabel("SDF")
plt.ylabel("Density")
plt.ylim(0, 1)
# plt.legend()
plt.show()