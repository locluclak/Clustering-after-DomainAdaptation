from scipy import stats
import matplotlib.pyplot as plt

# Sample sizes
nslist = [0.2, 0.4, 0.6, 0.8]

# Initialize FPR lists
FPRnaive = []
FPRpermutation = []
FPRpara = []
FPRoc = []
FPRnoinf = [1, 1, 1, 1]  # Example baseline (always 1)

# Define significance level
alpha = 0.05

# Read and compute FPR for each method
for ns in nslist:
    # Naive method
    with open(f"logs/selective_inference_log/FPR_naive_p_valueslist_rho{ns}.txt", "r") as f:
        pvalues = [float(line.strip()) for line in f if line.strip()]
    FPRnaive.append(sum(p < alpha for p in pvalues) / len(pvalues))
    
    # Permutation method
    with open(f"logs/selective_inference_log/FPRpermutation_p_valueslist_rho{ns}.txt", "r") as f:
        pvalues = [float(line.strip()) for line in f if line.strip()]
    FPRpermutation.append(sum(p < alpha for p in pvalues) / len(pvalues))
    
    # OC method
    with open(f"logs/selective_inference_log/FPR_oc_p_valueslist_rho{ns}.txt", "r") as f:
        pvalues = [float(line.strip()) for line in f if line.strip()]
    FPRoc.append(sum(p < alpha for p in pvalues) / len(pvalues))
    
    # Parametric method
    with open(f"logs/selective_inference_log/FPRpara_p_valueslist_rho{ns}.txt", "r") as f:
        pvalues = [float(line.strip()) for line in f if line.strip()]
    FPRpara.append(sum(p < alpha for p in pvalues) / len(pvalues))


print("FPR Naive:", FPRnaive)
print("FPR Permutation:", FPRpermutation)
print("FPR OC:", FPRoc)
print("FPR Parametric:", FPRpara)



# Plot FPR comparison
plt.figure(figsize=(8, 6))
plt.plot(nslist, FPRnoinf, 'k--', label='No Inference')
plt.plot(nslist, FPRnaive, 'o-', label='Naive')
plt.plot(nslist, FPRpermutation, 's-', label='Permutation')
plt.plot(nslist, FPRoc, '^-', label='OC')
plt.plot(nslist, FPRpara, 'd-', label='Parametric')
plt.ylim(0, 1)
plt.xlabel('correlation coefficient (rho)')
plt.ylabel('False Positive Rate (FPR)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('logs/selective_inference_log/FPR_comparison_rho.png')
plt.show()

