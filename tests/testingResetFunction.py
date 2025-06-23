''' script for roll over function'''
#%%
import numpy as np
import matplotlib.pyplot as plt

#%%

def resetFunction(flagX,y):
    y0= y[flagX]
    # no reset point
    if len(y0) == 0: return y
    dy0 = np.empty_like(y0)
    dy0[0] =y0[0]
    # more then one reset point
    if len(y0)>1: dy0[1:] = y0[1:]-y0[:-1]
    sy0 = np.zeros_like(y)
    sy0[flagX]= dy0
    bcg = np.cumsum(sy0)
    return y-bcg

#%% data generation

x = np.arange(100)
y = 3*x + 5

x0 = np.random.rand(len(x))>0.95
yReset = resetFunction(x0,y)


#%% data plot

fig, ax = plt.subplots()

ax.plot(x,y, 'r')
ax.plot(x[x0],y[x0],'o')
ax.plot(x,yReset,'b')
plt.show()

# %%
