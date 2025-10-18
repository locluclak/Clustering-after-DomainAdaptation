from scipy import stats
import matplotlib.pyplot as plt

# Sample sizes
nslist = [100, 150, 200, 250]

# Initialize FPR lists
FPRnaive = []
FPRpermutation = []
FPRpara = []
FPRoc = []
FPRbonf = []
FPRnoinf = [1, 1, 1, 1]  # Example baseline (always 1)

# Define significance level
alpha = 0.05

# Read and compute FPR for each method
# for ns in nslist:
#     # Naive method
#     with open(f"logs/selective_inference_log/FPRnaive_p_valueslist{ns}.txt", "r") as f:
#         pvalues = [float(line.strip()) for line in f if line.strip()]
#     FPRnaive.append(sum(p < alpha for p in pvalues) / len(pvalues))
#     FPRbonf.append(sum(p < alpha / 3 for p in pvalues) / len(pvalues))
    
#     # Permutation method
#     with open(f"logs/selective_inference_log/FPRpermutation_p_valueslist{ns}.txt", "r") as f:
#         pvalues = [float(line.strip()) for line in f if line.strip()]
#     FPRpermutation.append(sum(p < alpha for p in pvalues) / len(pvalues))
    
#     # OC method
#     with open(f"logs/selective_inference_log/FPRoc_p_valueslist{ns}.txt", "r") as f:
#         pvalues = [float(line.strip()) for line in f if line.strip()]
#     FPRoc.append(sum(p < alpha for p in pvalues) / len(pvalues))
    
#     # Parametric method
#     with open(f"logs/selective_inference_log/FPRpara_p_valueslist{ns}.txt", "r") as f:
#         pvalues = [float(line.strip()) for line in f if line.strip()]
#     FPRpara.append(sum(p < alpha for p in pvalues) / len(pvalues))


# for ns in nslist:
#     # Naive method
#     # with open(f"logs/selective_inference_log/TPRnaive_p_valueslist_delta{ns}.txt", "r") as f:
#     #     pvalues = [float(line.strip()) for line in f if line.strip()]
#     # FPRnaive.append(sum(p < alpha for p in pvalues) / len(pvalues))
    
#     # Permutation method
#     with open(f"logs/selective_inference_log/TPRpermutation_p_valueslist_delta{ns}.txt", "r") as f:
#         pvalues = [float(line.strip()) for line in f if line.strip()]
#     FPRpermutation.append(sum(p < alpha for p in pvalues) / len(pvalues))
    
#     # OC method
#     with open(f"logs/selective_inference_log/TPRoc_p_valueslist_delta{ns}.txt", "r") as f:
#         pvalues = [float(line.strip()) for line in f if line.strip()]
#     FPRoc.append(sum(p < alpha for p in pvalues) / len(pvalues))
    
#     # Parametric method
#     with open(f"logs/selective_inference_log/TPRpara_p_valueslist_delta{ns}.txt", "r") as f:
#         pvalues = [float(line.strip()) for line in f if line.strip()]
#     FPRpara.append(sum(p < alpha for p in pvalues) / len(pvalues))

# Plot FPR comparison
plt.figure(figsize=(8, 6))
plt.plot(nslist, FPRnoinf, 'k--', label='No Inference')
plt.plot(nslist, FPRnaive, 'o-', label='Naive')
plt.plot(nslist, FPRbonf, 'v-', label='Bonferroni')
plt.plot(nslist, FPRpermutation, 's-', label='Permutation')
plt.plot(nslist, FPRoc, '^-', label='OC')
plt.plot(nslist, FPRpara, 'd-', label='Parametric')
plt.ylim(0, 1.1)
plt.xlabel('Delta')
plt.ylabel('True Positive Rate (TPR)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('logs/selective_inference_log/TPR_comparison_delta.png')
plt.show()
