from collections import defaultdict
from mynlplib.clf_base import predict,make_feature_vector

# deliverable 4.1
def perceptron_update(x,y,weights,labels):
    '''
    compute the perceptron update for a single instance

    :param x: instance, a counter of base features and weights
    :param y: label, a string
    :param weights: a weight vector, represented as a dict
    :param labels: set of possible labels
    :returns: updates to weights, which should be added to weights
    :rtype: defaultdict

    '''

    update = defaultdict(float)
    y_hat, _ = predict(x, weights, labels)
    fv_true = defaultdict(float, make_feature_vector(x, y))
    fv_hat = defaultdict(float, make_feature_vector(x, y_hat))
    if y_hat is y:
        return update

    for fv_true_i in fv_true:
        update[fv_true_i] = fv_true[fv_true_i] - fv_hat[fv_true_i]
    
    for fv_hat_i in fv_hat:
        update[fv_hat_i] = fv_true[fv_hat_i] - fv_hat[fv_hat_i]

    return update

# deliverable 4.2
def estimate_perceptron(x,y,N_its):
    '''
    estimate perceptron weights for N_its iterations over the dataset (x,y)

    :param x: instance, a counter of base features and weights
    :param y: label, a string
    :param N_its: number of iterations over the entire dataset
    :returns: weight dictionary
    :returns: list of weights dictionaries at each iteration
    :rtype: defaultdict, list

    '''

    labels = set(y)
    weights = defaultdict(float)
    weight_history = []
    for it in range(N_its):
        for x_i,y_i in zip(x,y):
            # YOUR CODE GOES HERE
            update = perceptron_update(x_i,y_i,weights,labels)
            for u in update:
                weights[u] += update[u]
            
        weight_history.append(weights.copy())
    return weights, weight_history
