from scipy.stats import ttest_ind_from_stats

def get_stats(group_name):
    print(f"\nEnter values for {group_name}:")
    mean = float(input("  Mean: "))
    std = float(input("  Standard deviation: "))
    n = int(input("  Number of samples: "))
    return mean, std, n

print("Two-sample t-test (Welch) from summary statistics\n")

# Get stats for the two groups
mean1, std1, n1 = get_stats("Group 1")
mean2, std2, n2 = get_stats("Group 2")

# Compute t-test
t_stat, p_value = ttest_ind_from_stats(
    mean1=mean1, std1=std1, nobs1=n1,
    mean2=mean2, std2=std2, nobs2=n2,
    equal_var=False
)

# Compute pooled standard deviation
s_pooled = (( (n1 - 1) * std1**2 + (n2 - 1) * std2**2 ) / (n1 + n2 - 2)) ** 0.5

# Compute Cohen's d
cohens_d = (mean1 - mean2) / s_pooled

print(f"Cohen's d: {cohens_d:.4f}")

# Interpretation
if abs(cohens_d) < 0.2:
    interpretation = "negligible effect"
elif abs(cohens_d) < 0.5:
    interpretation = "small effect"
elif abs(cohens_d) < 0.8:
    interpretation = "medium effect"
elif abs(cohens_d) < 1.2:
    interpretation = "large effect"
else:
    interpretation = "very large effect"


# Print results
print("\n--- Results ---")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value:     {p_value:.6f}")

# Interpretation
alpha = 0.05
print("\n--- Conclusion ---")
if p_value < alpha:
    print("The p-value is below 0.05, so we reject the null hypothesis.")
    print("The two groups are likely not drawn from the same distribution.")
else:
    print("The p-value is above 0.05, so we cannot reject the null hypothesis.")
    print("The two groups are compatible with having been drawn from the same distribution.")
print(f"Effect size interpretation: {interpretation}")


# Scores between top pqk and classical svm:
# t-statistic: -2.0670
# p-value:     0.053608

# input values for best pqk, classical svm
# SVM: Score (95% confidence) = 0.892851 +/- 0.008748 == [0.884104, 0.901599]
# PQK: Score (95% confidence) = 0.900528 +/- 0.007837 == [0.892690, 0.908365]



