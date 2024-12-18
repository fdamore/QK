from itertools import permutations 

# to generate my_obs in the general non-local case. inputs are: string list, nr of qubits
# ['X','Y'], 3 --> ['XII', 'IXI', 'IIX', 'YII', 'IYI', 'IIY']
# ['XY'], 3 --> ['XZI', 'XIZ', 'ZXI', 'IXZ', 'ZIX', 'IZX']        
def generate_my_obs(pauli_str_list, n_qub):
    global_Id = 'I' * n_qub
    global_obs_list = []
    for string in pauli_str_list:
        # all possible dispositions of the selected obs (max one obs per qubit)
        obs_list = list(string)
        for combo in permutations(range(n_qub), len(obs_list)):
            global_obs = list(global_Id)
            for i, idx in enumerate(combo):
                global_obs[idx] = obs_list[i]
            global_obs_list.append(''.join(global_obs))
    # to avoid duplicates if there  are repeated obs (eg 2 X's)
    seen = set()
    unique_list = []
    for item in global_obs_list:
        if item not in seen:
            unique_list.append(item)
            seen.add(item)
    return unique_list

# to automatically generate strings of non local obs, involving adjacent qubits
# ['X'], 4, 2 --> ['XXII', 'IXXI', 'IIXX', 'XIIX']
def adjacent_qub_obs(single_pauli_list, n_qub, non_locality):
    global_Id = 'I' * n_qub
    global_obs_list = []
    for obs in single_pauli_list:
        for i in range(n_qub):
            global_obs = list(global_Id)
            for j in range(non_locality):
                global_obs[(i+j)%n_qub] = obs
            global_obs_list.append(''.join(global_obs))
    return global_obs_list
