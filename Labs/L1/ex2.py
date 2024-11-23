import matplotlib.pyplot as plt
import numpy as np

t1 = np.linspace(0, 2000, 1600)
x1 = np.sin(2 * np.pi * (1 / 400) * t1)
plt.figure()
plt.plot(t1, x1)
plt.show()

# b)
t2 = np.linspace(0, 3, 300)
x2 = np.sin(2 * np.pi * 800 * t2)
plt.figure()
plt.plot(t2, x2)
plt.show()

# c)
t3 = np.linspace(0, 0.3, 1600)
x3 = 300 * t3 % 10
plt.figure()
plt.plot(t3, x3)
plt.show()

# d)
t4 = np.linspace(0, 0.3, 5000)
x4 = np.sign(np.sin(t4 * 300))
plt.figure()
plt.plot(t4, x4)
plt.show()

# e)
t5 = np.random.rand(128, 128)
plt.figure()
plt.imshow(t5, cmap="viridis")
plt.show()

# f)
t6 = np.eye(128, 128)
for i in range(len(t6)):
    t6[i, i] = i
plt.figure()
plt.imshow(t6, cmap="plasma")
plt.show()