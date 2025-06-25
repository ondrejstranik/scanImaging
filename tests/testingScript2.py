''''
script to test  function processing flags
'''
#%%
import numpy as np
import matplotlib.pyplot as plt

#%%

lastCounter = 3

n = int(30)
x = np.arange(n)
y = (np.random.rand(n)> 0.5)*5


dy = np.diff(y,prepend=0)
flagY = dy>0
counter = lastCounter + np.cumsum(flagY)

fig, ax = plt.subplots()

ax.plot(x,y, color='red', label= 'y')
ax.plot(x,counter, color='red', label= 'counter')
ax.legend()
plt.show()






