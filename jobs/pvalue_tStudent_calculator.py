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
