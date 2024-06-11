import numpy as np
import gc
from sklearn.metrics.pairwise import cosine_similarity


def alpha_fe(orig, currents, status):
    alpha = 0.85

    alpha_total = 1.0
    weights = [1.0]
    for i in range(len(currents)):
        alpha_total *= alpha
        if status[i] == False:
            weights.append(0)
            continue
        
        weights.append(alpha_total)

    #print(weights)

    weights = np.array(weights) / np.sum(weights)

    return weights

def cosine_sim(orig, currents, status):
    sim_list = [np.exp(1)]
    for i in range(len(currents)):
        if status[i] == False:
            sim_list.append(0)
            continue
        
        similarity = 0.0

        for j in range(131):
            sim_value = np.exp(cosine_similarity(orig[j].reshape(1, -1), currents[i][j].reshape(1, -1)))[0][0]
            similarity += sim_value
            if i == 0:
                print(sim_value)

        sim_avg = similarity / float(131)

        sim_list.append(sim_avg)

    print(f"Sim list: {sim_list}")
    sim_list_np = np.array(sim_list)
    sim_list_scaled = sim_list_np / np.sum(sim_list_np)  

    print(f"Sim list scaled: {sim_list_scaled}")

    return sim_list_scaled

def euclidean_dist(orig, currents, status):
    dist_list = [0]
    for i in range(len(currents)):
        if status[i] == False:
            dist_list.append(np.nan)
            continue
        
        dist = np.linalg.norm(orig.flatten() - currents[i].flatten())
        dist_list.append(dist)

    dist_list_np = np.array(dist_list) 
    max_dist = np.nanmax(dist_list_np)
    if np.isnan(max_dist):
        return orig
    
    dist_list_np /= max_dist

    exp_dist_list_np = np.exp(-dist_list_np)

    exp_dist_list_np_without_nan = np.nan_to_num(exp_dist_list_np, nan=0)

    #print(f"exp_dist_list_np_without_nan:\n {exp_dist_list_np_without_nan}")

    scaled_weight = exp_dist_list_np_without_nan / np.sum(exp_dist_list_np_without_nan)

    #print(f"Scaled weight:\n {scaled_weight}")

    return scaled_weight
    

if __name__ == "__main__":
    orig = np.random.randn(131, 5, 5)
    currents = []
    status = [True for i in range(5)]
    for i in range(5):
        currents.append(np.random.randn(131, 5, 5))
    cosine_sim(orig, currents, status)