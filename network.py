"""Class that represents the network to be evolved."""
import random
import logging
from train import train_and_score, train_and_score_TB

class Network():
    """Represent a network and let us operate on it.

    Modified to work for CNN.
    """

    def __init__(self, nn_param_choices=None):
        """Initialize our network.

        Args:
            nn_param_choices (dict): Parameters for the network, includes:
                filter_size (list): [(3,3), (5,5), (7,7)],
                batch_size (list): [10, 20, 30, 40, 50],
                l1_penalty (list): [0, 1e-1, 1e-2, 1e-3, 1e-4],
                l2_penalty (list): [0, 1e-1, 1e-2, 1e-3, 1e-4],
                learning_rate (list): [1e-1, 1e-2, 1e-3],
                conv_layer_count (list): [1,2],
                filters_per_conv (list): [x for x in range(10,51)],
                hidden_layer_count (list): [1,2,3],
                units_per_hidden (list): [x for x in range(50,501)]
        """
        self.accuracy = 0.
        self.nn_param_choices = nn_param_choices
        self.network = {}  # (dic): represents CNN network parameters

    def create_random(self):
        """Create a random network."""
        for key in self.nn_param_choices:
            self.network[key] = random.choice(self.nn_param_choices[key])

    def create_set(self, network):
        """Set network properties.

        Args:
            network (dict): The network parameters

        """
        self.network = network

    def train(self, dataset, iteration, current_network_count):
        """Train the network and record the accuracy.

        Args:
            dataset (str): Name of dataset to use.

        """
        if self.accuracy == 0.:
            self.accuracy = train_and_score_TB(self.network, dataset, iteration, current_network_count)

    def print_network(self):
        """Print out a network."""
        logging.info(self.network)
        logging.info("Network accuracy: %.2f%%" % (self.accuracy * 100))
	
    def get_parameters(self):
        # Returns parameters of choice.
        return self.network
