delta_list = [10]

for delta in delta_list:
    with open(f"logs/selective_inference_log/TPR_teststatlist_rotated_delta{delta}.txt", "r") as f:
        lines = f.readlines()
        teststat_list = [float(line.strip()) for line in lines]
    mean_teststat = sum(teststat_list) / len(teststat_list)
    print(f"Delta: {delta}, Mean Test Statistic: {mean_teststat}")


# Delta: 2, Mean Test Statistic: 2.64227636824476
# Delta: 4, Mean Test Statistic: 3.677936269211288
# Delta: 6, Mean Test Statistic: 4.824802597223915
# Delta: 8, Mean Test Statistic: 6.4136899414452495
# Delta: 10, Mean Test Statistic: 8.003660877890127

# dim 20 
# Delta: 10, Mean Test Statistic: 8.361759163824727
# Delta: 8, Mean Test Statistic: 7.081944094935953
# Delta: 6, Mean Test Statistic: 5.120661088816247
# Delta: 4, Mean Test Statistic: 3.993231013975166
# Delta: 2, Mean Test Statistic: 3.383460275550365