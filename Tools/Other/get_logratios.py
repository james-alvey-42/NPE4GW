import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

lrs = pd.read_pickle('/Users/knana/Documents/GitHub/NPE4GW/Tools/Other/logratios_R1 (1)')

#plot histogram of logratios
plt.figure(figsize=(15, 8))
for idx in range(len(lrs.parnames)):
    ax = plt.subplot(5, 3, idx + 1)
    logratios = lrs.logratios[:, idx]
    params = lrs.params[:, idx, 0]
    plt.hist(params, weights=np.exp(logratios.numpy()), bins=100)
plt.show()
