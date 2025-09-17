
from scipy import stats
import matplotlib.pyplot as plt
with open("logs/selective_inference_log/p_values.txt","r") as f:
    pvalues = [float(line.strip()) for line in f if line.strip()]

# Significance threshold
alpha = 0.05

# Count how many p-values are below threshold
num_rejected = sum(p < alpha for p in pvalues)

# Total number of tests
num_tests = len(pvalues)

# False Positive Rate
fpr = num_rejected / num_tests
print("FPR:",fpr)
# Kiểm định thống kê
kstest = stats.kstest(pvalues, "uniform")
print(kstest)
plt.hist(pvalues)
plt.show()
