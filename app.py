# Adding Q-Q Plot Visualization

import matplotlib.pyplot as plt
import scipy.stats as stats

# Normality check section code...
# Add your code for normality check here

# Q-Q plot for the original population distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
stats.probplot(original_population, dist="norm", plot=plt)
plt.title('Q-Q Plot: Original Population')

# Q-Q plot for sample means distribution
plt.subplot(1, 2, 2)
stats.probplot(sample_means, dist="norm", plot=plt)
plt.title('Q-Q Plot: Sample Means')

plt.tight_layout()
plt.show()