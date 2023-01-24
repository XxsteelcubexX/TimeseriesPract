%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

# number of time steps
T = 1000

#intial Price
P0 = 100

#drift
mu = 0.001


# ███████ ██ ███    ███ ██    ██ ██       █████  ████████ ██  ██████  ███    ██
# ██      ██ ████  ████ ██    ██ ██      ██   ██    ██    ██ ██    ██ ████   ██
# ███████ ██ ██ ████ ██ ██    ██ ██      ███████    ██    ██ ██    ██ ██ ██  ██
#      ██ ██ ██  ██  ██ ██    ██ ██      ██   ██    ██    ██ ██    ██ ██  ██ ██
# ███████ ██ ██      ██  ██████  ███████ ██   ██    ██    ██  ██████  ██   ████


last_p = np.log(P0)
log_returns = np.zeros(T)
prices = np.zeros(T)

for t in range(T):
    #sample a log return
    r = 0.01 * np.random.randn() # we are adding random noise here

    #compute the new log price
    p = last_p + mu + r

    #store  the return and price
    log_returns[t] = r + mu
    prices[t] = np.exp(p)

    #assign last p
    last_p = p

plt.figure(figsize = (20,8))
plt.plot(prices)
plt.title('Simulated Prices')
# vectorize  the above loop
