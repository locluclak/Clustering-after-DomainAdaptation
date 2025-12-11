from scipy import stats
import matplotlib.pyplot as plt

# Sample sizes
nslist = [4,6,8,10]
# nslist = [100,150,200,250]

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
#     FPRbonf.append(sum(p < alpha / (3**100) for p in pvalues) / len(pvalues))
    
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


for ns in nslist:
    # Bonferroni method
    with open(f"logs/selective_inference_log/TPRnaive_p_valueslist_delta{ns}.txt", "r") as f:
        pvalues = [float(line.strip()) for line in f if line.strip()]
    FPRbonf.append(sum(p < alpha / (3**100) for p in pvalues) / len(pvalues))

    
    # Permutation method
    # with open(f"logs/selective_inference_log/TPRpermutate_p_valueslist_delta{ns}.txt", "r") as f:
    #     pvalues = [float(line.strip()) for line in f if line.strip()]
    # FPRpermutation.append(sum(p < alpha for p in pvalues) / len(pvalues))
    
    # OC method
    with open(f"logs/selective_inference_log/TPRoc_p_valueslist_delta{ns}.txt", "r") as f:
        pvalues = [float(line.strip()) for line in f if line.strip()]
    FPRoc.append(sum(p < alpha for p in pvalues) / len(pvalues))
    
    # Parametric method
    with open(f"logs/selective_inference_log/TPRparametric_p_valueslist_delta{ns}.txt", "r") as f:
        pvalues = [float(line.strip()) for line in f if line.strip()]
    FPRpara.append(sum(p < alpha for p in pvalues) / len(pvalues))

# Plot FPR comparison
plt.figure(figsize=(8, 6))
# Set the explicit tick locations for the x-axis
plt.xticks(nslist)

# plt.plot(nslist, FPRnaive, 'o-', label='Naive')
# plt.plot(nslist, FPRnoinf, 'k--', label='No Inference')
plt.plot(nslist, FPRbonf, 'v-', label='Bonferroni')
# plt.plot(nslist, FPRpermutation, 's-', label='Permutation')
plt.plot(nslist, FPRoc, '^-', label='OC')
plt.plot(nslist, FPRpara, 'd-', label='Parametric')
plt.ylim(-0.01, 1.1)
plt.xlabel('delta')
# plt.xlabel('# source instances')
# plt.ylabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.savefig('logs/selective_inference_log/FPR_comparison_sample_size.pdf')
plt.savefig('logs/selective_inference_log/TPR_comparison_delta.pdf')
plt.show()
