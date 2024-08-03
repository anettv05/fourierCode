
import numpy as np
import matplotlib.pyplot as plt

kp=10*np.pi
kf=2*np.pi*10**5
fm=5000
t = np.linspace(0,10,2000)
mt = np.sign(np.sin(2*fm*np.pi*t))
wc = 2* np.pi * 10**7
amp = np.sin(wc*t+kp*mt)
imt = np.cumsum(mt)
amp2 = np.sin(wc*t+kf*imt)
wi1 = np.gradient(wc*t+mt, t)
wi2 = np.gradient(wc*t+imt, t)


plt.figure(figsize=(10,6))
plt.subplot(4,1,1)
plt.plot(t,amp)
plt.title("Phase modulation")
plt.xlabel("time")
plt.ylabel("amp")

plt.subplot(4,1,3)
plt.plot(t,amp2)
plt.title("Freq modulation")
plt.xlabel("time")
plt.ylabel("amp")

plt.subplot(4,1,2)
plt.plot(t,wi1)
plt.title("Phase modulation inst freq")
plt.xlabel("time")
plt.ylabel("freq")

plt.subplot(4,1,4)
plt.plot(t,wi2)
plt.title("Freq modulation inst freq")
plt.xlabel("time")
plt.ylabel("freq")
plt.tight_layout()
plt.show()
