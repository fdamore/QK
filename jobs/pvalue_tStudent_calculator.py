import numpy as np
from scipy.stats import ttest_ind, ttest_rel
import os

def get_stats_group(group_name):
    print(f"\nEnter values for {group_name}:")
    mean = float(input("  Mean: "))
    std = float(input("  Standard deviation: "))
    n = int(input("  Number of samples: "))
    return mean, std, n

def load_fold_array(file_path):
    """
    Carica valori da file con due possibili offset:
    - se il file inizia con '2' -> partire dalla riga 18
    - se il file inizia con '0' o '1' -> partire dalla riga 16
    Formato atteso:
    Fold X: valore
    Stampa anche i valori letti e il numero di fold.
    """
    filename = os.path.basename(file_path)
    first_char = filename[0]

    if first_char == '2':
        start_line = 17  # line 18
    elif first_char in ['0', '1']:
        start_line = 15  # line 16
    else:
        raise ValueError(f"Filename {filename} must start with '0', '1', or '2'!")

    with open(file_path, "r") as f:
        lines = f.readlines()[start_line:]

    arr = []
    for line in lines:
        try:
            value = float(line.strip().split(":")[1].strip())
            arr.append(value)
        except:
            pass  # ignora eventuali righe malformate

    arr = np.array(arr)
    print(f"\nLoaded {len(arr)} folds from {filename}:")
    for i, val in enumerate(arr, start=1):
        print(f"Fold {i}: {val}")
    return arr

def compute_cohens_d_independent(mean1, std1, n1, mean2, std2, n2):
    s_pooled = (( (n1 - 1) * std1**2 + (n2 - 1) * std2**2 ) / (n1 + n2 - 2)) ** 0.5
    d = (mean1 - mean2) / s_pooled
    return d

def compute_cohens_d_paired(diff_array):
    d = np.mean(diff_array) / np.std(diff_array, ddof=1)
    return d

# --- Main program ---
print("Choose test type:")
print("1 - Independent t-test")
print("2 - Paired t-test")
choice = input("Enter 1 or 2: ").strip()

if choice == "1":
    # Independent t-test
    mean1, std1, n1 = get_stats_group("Group 1")
    mean2, std2, n2 = get_stats_group("Group 2")
    
    from scipy.stats import ttest_ind_from_stats
    t_stat, p_value = ttest_ind_from_stats(
        mean1=mean1, std1=std1, nobs1=n1,
        mean2=mean2, std2=std2, nobs2=n2,
        equal_var=False
    )
    
    cohens_d = compute_cohens_d_independent(mean1, std1, n1, mean2, std2, n2)
    
elif choice == "2":
    # Paired t-test
    path1 = input("Enter path to file for Group 1: ").strip()
    path2 = input("Enter path to file for Group 2: ").strip()
    
    arr1 = load_fold_array(path1)
    arr2 = load_fold_array(path2)
    
    if len(arr1) != len(arr2):
        raise ValueError("Arrays must have the same length for paired test!")
    
    diff = arr1 - arr2
    
    t_stat, p_value = ttest_rel(arr1, arr2)
    cohens_d = compute_cohens_d_paired(diff)
    
else:
    raise ValueError("Invalid choice. Enter 1 or 2.")

# --- Print results ---
print("\n--- Results ---")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value:     {p_value:.6f}")
print(f"Cohen's d:  {cohens_d:.4f}")

# Interpretation
alpha = 0.05
print("\n--- Conclusion ---")
if p_value < alpha:
    print("The p-value is below 0.05, so we reject the null hypothesis.")
    print("The two groups are likely not drawn from the same distribution.")
else:
    print("The p-value is above 0.05, so we cannot reject the null hypothesis.")
    print("The two groups are compatible with having been drawn from the same distribution.")

# Cohen's d interpretation
abs_d = abs(cohens_d)
if abs_d < 0.2:
    effect_interpretation = "negligible effect"
elif abs_d < 0.5:
    effect_interpretation = "small effect"
elif abs_d < 0.8:
    effect_interpretation = "medium effect"
elif abs_d < 1.2:
    effect_interpretation = "large effect"
else:
    effect_interpretation = "very large effect"

print(f"Effect size interpretation: {effect_interpretation}")



# INDEPENDENT TEST
# SVM vs best PQK
# SVM: Score (95% confidence) = 0.892851 +/- 0.008748 == [0.884104, 0.901599]
# PQK: Score (95% confidence) = 0.900528 +/- 0.007837 == [0.892690, 0.908365]
# t-statistic: 2.0733
# p-value:     0.052937
# Cohen's d:  0.9272


################ PAIRED TEST ###############
######### SVM vs best PQK
# selected_scores/0_SVM_CLASSIC.accuracy.txt
# selected_scores/1_PQK_M2_3D_ENT_TRUE_18obs.accuracy.txt
# t-statistic: 1.9056
# p-value:     0.089081
# Cohen's d:   0.6026

# selected_scores/0_SVM_CLASSIC.f1.txt.txt
# selected_scores/1_PQK_M2_3D_ENT_TRUE_18obs.f1.txt
# t-statistic: -2.7063
# p-value:     0.024142
# Cohen's d:  -0.8558


######### best QK vs best PQK
# selected_scores/2_QK_TROTTER_entFalse.accuracy.txt
# selected_scores/1_PQK_M2_3D_ENT_TRUE_18obs.accuracy.txt
# t-statistic: 4.2241
# p-value:     0.002226
# Cohen's d:   1.3358

# selected_scores/1_PQK_M2_3D_ENT_TRUE_18obs.f1.txt
# selected_scores/2_QK_TROTTER_entFalse.f1.txt
# t-statistic: 6.4534
# p-value:     0.000118
# Cohen's d:  2.0407