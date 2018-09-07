"""Iterate over every combination of hyperparameters."""
import logging
from network import Network
from tqdm import tqdm
import random
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='random-log.txt'
)

def train_networks(networks, dataset):
    """Train each network.

    Args:
        networks (list): Current population of networks
        dataset (str): Dataset to use for training/evaluating
    """
    pbar = tqdm(total=len(networks))
    for network in networks:
        network.train(dataset)
        network.print_network()
        pbar.update(1)
    pbar.close()

    # Sort our final population.
    networks = sorted(networks, key=lambda x: x.accuracy, reverse=True)

    # Print out the top 5 networks.
    # print_networks(networks[:5])
    
    # Also return the best 5 networks.
    return networks[:5]

def print_networks(networks):
    """Print a list of networks.

    Args:
        networks (list): The population of networks

    """
    logging.info('-'*80)
    for network in networks:
        network.print_network()

def generate_network_list(nn_param_choices, population):
    """Generate a list of random networks.

    Args:
        nn_param_choices (dict): The parameter choices
        population (int): number of CNNs to generate

    Returns:
        networks (list): A list of network objects

    """
    networks = []
    
    for i in range(population):
        network = {}
        for key in nn_param_choices:
            network[key] = random.choice(nn_param_choices[key])
            
        # Instantiate a network object with set parameters.
        network_obj = Network()
        network_obj.create_set(network)

        networks.append(network_obj)

    return networks

def main():
    """Brute force test every network."""
    dataset = 'cifar10'
    generations = 10  # Number of iterations
    population = 20  # Number of networks in each generation.
    selected_networks = []

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

    logging.info("***Random sampling networks***")
    
    for i in range(generations):
        logging.info("***Doing generation %d of %d***" %
                     (i + 1, generations))
        networks = generate_network_list(nn_param_choices, population)
        selected_networks.extend(train_networks(networks, dataset))
    
    selected_networks = sorted(selected_networks, key=lambda x: x.accuracy, reverse=True)
    
    # Print out the top 5 networks.
    print_networks(selected_networks[:5])

if __name__ == '__main__':
    main()
