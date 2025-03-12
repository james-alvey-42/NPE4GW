import pandas as pd
import matplotlib.pyplot as plt

lrs = pd.read_pickle('/Users/knana/Documents/GitHub/NPE4GW/Tools/Other/logratios_R1 (1)')
print(lrs.logratios.shape)

#plot histogram of logratios
plt.figure(figsize=(10, 5))
plt.title("TMNRE logratios")
plt.hist(lrs.logratios[:,2], bins=30, density=True, alpha=0.7, label="logratios")
plt.xlabel("Chirp mass")
plt.ylabel("Density")
plt.legend()
plt.show()
