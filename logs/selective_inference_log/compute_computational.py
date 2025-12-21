import math

def calculate_stats(filename="data.txt"):
    with open(filename, 'r') as f:
        data = [float(line.strip()) for line in f if line.strip()]
    
    n = len(data)
    if n == 0:
        return "File rỗng."
    
    mean = sum(data) / n
    
    variance = sum((x - mean) ** 2 for x in data) / n
    std_dev = math.sqrt(variance)
    
    return f"{mean:.4f} \pm {std_dev:.4f}"

print(calculate_stats("logs/selective_inference_log/countitvtpr_249.txt"))

# time
# 34.7623 \pm 3.5437
# 49.2781 \pm 6.9362
# 52.7587 \pm 11.4290
# 62.6801 \pm 13.6769

# itv
# 1290.1739 \pm 512.7404
# 1501.0177 \pm 415.2149
# 1503.8053 \pm 397.9404
# 1489.7523 \pm 426.0768