from numpy import matmul
from itertools import product
from numpy import inf

def initial_state_probability(pi_vector, transition_matrix, limit):
    """Returns probability matrix of inicialisation for each state in transition matrix
      using pi_vector as starting reference

    Args:
        pi_vector (array): 0s & 1s array, 1=started state, 0=inactive state during this cycle
        transition_matrix (2D array): array of probabilities of transitions between all states
        limit (int): number of cycles before shutdown to prevent infinity problem

    Returns:
        array: List of probabilities of each state to be the
          starting point for the sequence.
    """
    flag = False
    counter =0

    while not flag:
        res_matrix = matmul(pi_vector,transition_matrix) # calculate new matrix
        
        #round up values to avoid unecessary deep loops
        formatted_matrix = [ round(elem,8) for elem in res_matrix ]

        if all(x==y for x,y in zip(pi_vector,formatted_matrix)):
            return res_matrix
        
        elif counter >= limit:
            print(f"[!] Limit Reached: {limit}, cycles...")
            return formatted_matrix
        
        #update pi_vector for next cycle
        else: pi_vector = formatted_matrix

        counter +=1
    

def sequence_probability(obs_transition_matrix, hid_transition_matrix, current_hidden_state_lst, current_observation):
    
    probabilities = []

    for i in range(len(current_observation)):
        
        #probability of hidden states based on history
        cur_hid_loc = current_hidden_state_lst[i]
        if i != 0:    
            prev_hid_loc = current_hidden_state_lst[i-1]
            probabilities.append(hid_transition_matrix[cur_hid_loc][prev_hid_loc])
        
        else:
            #first save starting state probability (most complicated method)
            state_filter = [0]*len(hid_transition_matrix[0]) # make list of possible options based on length of hidden transition matrix row
            
            #activate only the position corresponding to the current_hiddent_state_list array info (using 1 as indicator, rest are marked as 0)
            state_filter[current_hidden_state_lst[0]] = 1

            prob_list = initial_state_probability(state_filter,
                                                  hid_transition_matrix,200)
            
            probabilities.append(prob_list[current_hidden_state_lst[0]])

        #obstain probability of observed state with hidden state
        probabilities.append(obs_transition_matrix[cur_hid_loc][current_observation[i]])

    #multipy all probabilities to get final estimation
    result = 1
    for element in probabilities:
        result *= element
    
    return round (result,3)


hidden = [
    [0.5,0.3,0.2],
    [0.4,0.2,0.4],
    [0.0,0.3,0.7]
]

observed = [
    [0.9,0.1],
    [0.6,0.4],
    [0.2,0.8]
]

observation = [1,1,0]


combinations = list(product(range(len(hidden[0])),repeat=len(observation)))
#get all possible combinations of the hidden_observation_array [0,2,1] -> [0,0,1]->...etc

final = -inf
best_combo = None
for combo in combinations:
    res = sequence_probability(observed,hidden,combo,observation)

    if res>final:
        final = res
        best_combo = combo

print(f"For state {observation}\nMost likely combination: {best_combo}, with a {final*100}% probability.")

# if the input is a single one just put a for loop for the initial_state_probability method
