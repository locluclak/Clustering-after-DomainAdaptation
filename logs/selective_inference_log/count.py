
from scipy import stats
import matplotlib.pyplot as plt
nslist = [100,150,200,250]

FPR = []
for ns in nslist:
    with open(f"logs/selective_inference_log/FPRpermutation_p_valueslist{ns}.txt","r") as f:
        pvalues = [float(line.strip()) for line in f if line.strip()]

    # Significance threshold
    alpha = 0.05

    # Count how many p-values are below threshold
    num_rejected = sum(p < alpha for p in pvalues)

    # Total number of tests
    num_tests = len(pvalues)

    # False Positive Rate
    fpr = num_rejected / num_tests
    FPR.append(fpr)
    print("FPR:",fpr)

# Kiểm định thống kê
# kstest = stats.kstest(pvalues, "uniform")
# print(kstest)
# plt.hist(pvalues)
# plt.show()
