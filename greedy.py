"""Iterate over every combination of hyperparameters."""
import logging
from network import Network
from tqdm import tqdm

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='greedy-log.txt'
)

def greedy(network, dataset, nn_param_choices, tolerance):
    """Greedily evolves a network.
    
    Args:
        network (Network): a seed network to start with.
        dataset (str): Dataset to use for training/evaluating
        nn_param_choices (dict): The parameter choices
        tolerance (int): Stop after no improvement for "tolerant" iterations.
        
    Returns:
        best_network (Network): The network with the best performance.

    """
    evolve_counter - 0
    best_network - Network()
    best_acc = 0
    elapsed_time = 0
    
    # train the very first instance.
    network.train(dataset)
    network.print_network()
    
    while evolve_counter < tolerance:
        print("Running on the",elapsed_time,"iterations(s).")
        if network.accuracy > best_acc:
            # switch to new configuration, if it's better.
            best_network = network
            best_acc = network.accuracy
            evolve_counter = 0
        else:
            # stay on the current one.
            evolve_counter += 1
            print("Staying with the current best performance network. Evolve count:",evolve_counter)
        
        # Randomly change one parameter at a time.
        key = random.choice(list(nn_param_choices.keys()))
        val = random.choice(nn_param_choices[key])
        while val == network[key]:
            val = random.choice(nn_param_choices[key])
        network[key] = val
        network.train(dataset)
        network.print_network()
        elapsed_time += 1

    return best_network

def main():
    """Greedy approach."""
    dataset = 'cifar10'

    nn_param_choices = {
        'batch_size':[10, 20, 30, 40, 50],
		'hidden_layer_count':[1,2,3],
        'units_per_hidden':[x for x in range(50,550,50)],
        'l1_penalty':[0, 1e-1, 1e-2, 1e-3, 1e-4],
        'l2_penalty':[0, 1e-1, 1e-2, 1e-3, 1e-4],
        'learning_rate':[1e-1, 1e-2, 1e-3],
        'conv_layer_count':[1,2],
        'filters_per_conv':[x for x in range(10,60,10)],
		'filter_size':[(3,3), (5,5), (7,7)],
    }

    logging.info("***Greedy approach***")
    
    # First, get a random configuration.
    for key in nn_param_choices:
        seed_param[key] = random.choice(nn_param_choices[key])
        
    seed = Network()
    seed.create_set(seed_param)
    
    # setting the tolerance = 5; if there's no improvement on accuracy after 5 iterations, assumed that we've reached the local maximum.
    best_network = greedy(seed, dataset, nn_param_choices, 5)

if __name__ == '__main__':
    main()
