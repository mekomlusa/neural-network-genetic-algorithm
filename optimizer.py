"""
Class that holds a genetic algorithm for evolving a network.

Credit:
    A lot of those code was originally inspired by:
    http://lethain.com/genetic-algorithms-cool-name-damn-simple/
"""
from functools import reduce
from operator import add
import random
from network import Network
import operator

class Optimizer():
    """Class that implements genetic algorithm for CNN optimization."""

    def __init__(self, nn_param_choices, retain=0.4,
                 random_select=0.1, mutate_chance=0.1):
        """Create an optimizer.

        Args:
            nn_param_choices (dict): Possible network paremters
            retain (float): Percentage of population to retain after
                each generation
            random_select (float): Probability of a rejected network
                remaining in the population
            mutate_chance (float): Probability a network will be
                randomly mutated

        """
        self.mutate_chance = mutate_chance
        self.random_select = random_select
        self.retain = retain
        self.nn_param_choices = nn_param_choices

    def create_population(self, count):
        """Create a population of random networks.

        Args:
            count (int): Number of networks to generate, aka the
                size of the population

        Returns:
            (list): Population of network objects

        """
        pop = []
        for _ in range(0, count):
            # Create a random network.
            network = Network(self.nn_param_choices)
            network.create_random()

            # Add the network to our population.
            pop.append(network)

        return pop

    @staticmethod
    def fitness(network):
        """Return the accuracy, which is our fitness function."""
        return network.accuracy

    def grade(self, pop):
        """Find average fitness for a population.

        Args:
            pop (list): The population of networks

        Returns:
            (float): The average accuracy of the population

        """
        summed = reduce(add, (self.fitness(network) for network in pop))
        return summed / float((len(pop)))

    def breed(self, mother, father):
        """Make two children as parts of their parents.

        Args:
            mother (dict): Network parameters
            father (dict): Network parameters

        Returns:
            (list): Two network objects

        """
        children = []
        
        all_param = self.nn_param_choices
        del all_param['conv_layer_count']
        del all_param['filter_size']
        del all_param['filters_per_conv']
        del all_param['hidden_layer_count']
        del all_param['units_per_hidden']
        
        for i in range(2):

            child = {}
            
            # Recombine some parameters, as specified in the paper.
            # Only decide what conv and hidden layer parameters are exchanged at the very first run
            if i == 0:
                default_cov_list = ['m','f']
                default_hid_list = ['m','f']
                cov_choice = random.choice(['m','f'])
                hid_choice = random.choice(['m','f'])
            
            if cov_choice == 'm':
                # covlutional layers will choose from the mother network
                child['conv_layer_count'] = mother.network['conv_layer_count']
                child['filter_size'] = mother.network['filter_size']
                child['filters_per_conv'] = mother.network['filters_per_conv']
            else:
                # covlutional layers will choose from the father network
                child['conv_layer_count'] = father.network['conv_layer_count']
                child['filter_size'] = father.network['filter_size']
                child['filters_per_conv'] = father.network['filters_per_conv']
                
            if hid_choice == 'm':
                # hidden layers will choose from the mother network
                child['hidden_layer_count'] = mother.network['hidden_layer_count']
                child['units_per_hidden'] = mother.network['units_per_hidden']
            else:
                # hidden layers will choose from the father network
                child['hidden_layer_count'] = father.network['hidden_layer_count']
                child['units_per_hidden'] = father.network['units_per_hidden']

            # Loop through the parameters and pick params for the kid.
            for param in all_param:
                child[param] = random.choice(
                    [mother.network[param], father.network[param]]
                )

            # Now create a network object.
            network = Network(self.nn_param_choices)
            network.create_set(child)

            # Randomly mutate some of the children.
            if self.mutate_chance > random.random():
                network = self.mutate(network)

            children.append(network)
            
            # Switch the choice so that it works for the next child.
            if i == 0:
                default_cov_list.remove(cov_choice)
                default_hid_list.remove(hid_choice)
                cov_choice = default_cov_list[0]
                hid_choice = default_hid_list[0]

        return children

    def mutate(self, network):
        """Randomly mutate one part of the network.

        Args:
            network (dict): The network parameters to mutate

        Returns:
            (Network): A randomly mutated network object

        """
        # If selected to be mutated, each parameter is subjected 
        # to change based on the prob. specified in the paper
        
        # "the number of the convolutional or hidden layers is
        # in- or decreased by one with a probability of 50% each"
        layer_inc_chance = random.random()
        selected_layers = random.choice(['conv_layer_count','hidden_layer_count'])
        if layer_inc_chance >= 0.5:
            network.network[selected_layers] += 1
        else:
            network.network[selected_layers] -= 1
            
        rate_chance = random.random()
        if rate_chance <= 0.1:
            network.network['learning_rate'] = network.network['learning_rate']*random.choice([100,0.01])
            if network.network['l1_penalty'] >= 1e-5 and network.network['l2_penalty'] >= 1e-5:
                # follow the same rule here
                network.network['l1_penalty'] = network.network['l1_penalty']*random.choice([100,0.01])
                network.network['l2_penalty'] = network.network['l2_penalty']*random.choice([100,0.01])
        elif rate_chance <= 0.4:
            network.network['learning_rate'] = network.network['learning_rate']*10
            if network.network['l1_penalty'] >= 1e-5 and network.network['l2_penalty'] >= 1e-5:
                # follow the same rule here
                network.network['l1_penalty'] = network.network['l1_penalty']*10
                network.network['l2_penalty'] = network.network['l2_penalty']*10
        elif rate_chance <= 0.9:
            network.network['learning_rate'] = network.network['learning_rate']*0.01
            if network.network['l1_penalty'] >= 1e-5 and network.network['l2_penalty'] >= 1e-5:
                # follow the same rule here
                network.network['l1_penalty'] = network.network['l1_penalty']*0.01
                network.network['l2_penalty'] = network.network['l2_penalty']*0.01
                
        # Special treatment for 0 l1 and l2 penalties
        if network.network['l1_penalty'] == 0:
            increase_chance = random.random()
            if increase_chance <= 0.4:
                network.network['l1_penalty'] = random.choice([0.001, 0.0001])
        elif network.network['l1_penalty'] < 1e-5:
            network.network['l1_penalty'] = 0
            
        if network.network['l2_penalty'] == 0:
            increase_chance = random.random()
            if increase_chance <= 0.4:
                network.network['l2_penalty'] = random.choice([0.001, 0.0001])
        elif network.network['l2_penalty'] < 1e-5:
            network.network['l2_penalty'] = 0
        
        # Mutating number of filters and hidden units, filter size, and batch size here
        unit_chance = random.random()
        if unit_chance <= 0.1:
            # Change number of filters only if the result is still positive.
            num_filter_after = network.network['filters_per_conv'] + random.choice([20,-20])
            if num_filter_after > 0:
                network.network['filters_per_conv'] = num_filter_after
            # Change number of hidden units only if the result is still positive.
            hidden_unit_after = network.network['units_per_hidden'] + random.choice([100,-100])
            if hidden_unit_after > 0:
                network.network['units_per_hidden'] = hidden_unit_after
            # Change filter size only if the result is still positive.
            filter_size_after = tuple(map(operator.add, network.network['filter_size'],random.choice([(-4,-4),(4,4)])))
            if sum(filter_size_after) > 0:
                network.network['filter_size'] = filter_size_after
            # Change batch size only if the result is still positive
            batch_size_after = network.network['batch_size'] + random.choice([20,-20])
            if batch_size_after > 0:
                network.network['batch_size'] = batch_size_after
        elif unit_chance <= 0.5:
            # Change number of filters only if the result is still positive.
            num_filter_after = network.network['filters_per_conv'] + random.choice([10,-10])
            if num_filter_after > 0:
                network.network['filters_per_conv'] = num_filter_after
            # Change number of hidden units only if the result is still positive.
            hidden_unit_after = network.network['units_per_hidden'] + random.choice([50,-50])
            if hidden_unit_after > 0:
                network.network['units_per_hidden'] = hidden_unit_after
            # Change filter size only if the result is still positive.
            filter_size_after = tuple(map(operator.add, network.network['filter_size'],random.choice([(-2,-2),(2,2)])))
            if sum(filter_size_after) > 0:
                network.network['filter_size'] = filter_size_after
            # Change batch size only if the result is still positive
            batch_size_after = network.network['batch_size'] + random.choice([10,-10])
            if batch_size_after > 0:
                network.network['batch_size'] = batch_size_after

        return network

    def evolve(self, pop):
        """Evolve a population of networks.

        Args:
            pop (list): A list of network parameters

        Returns:
            (list): The evolved population of networks

        """
        # Get scores for each network.
        graded = [(self.fitness(network), network) for network in pop]

        # Sort on the scores.
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]
        
        # First, keep the best 3 CNNs.
        parents = graded[:3]
        
        # Randomly created 3 new CNNs and added to the next generations.
        for _ in range(0, 3):
            network = Network(self.nn_param_choices)
            network.create_random()
            parents.append(network)
            
        # Need to get 44 parents randomly
        cutoff = int(len(graded)*0.75)
        parents_length = len(parents)
        desired_length = len(pop) - parents_length
        children = []

        # Add children, which are bred from the remaining networks.
        while len(children) < desired_length:

            # Get a random mom and dad.
            if random.random() <= 0.75:
                male = random.randint(0, cutoff)
            else:
                male = random.randint(cutoff+1, len(graded)-1)
                
            if random.random() <= 0.75:
                female = random.randint(0, cutoff)
            else:
                female = random.randint(cutoff+1, len(graded)-1)

            # Assuming they aren't the same network...
            if male != female:
                male = graded[male]
                female = graded[female]

                # Breed them.
                babies = self.breed(male, female)

                # Add the children one at a time.
                for baby in babies:
                    # Don't grow larger than desired length.
                    if len(children) < desired_length:
                        children.append(baby)

        parents.extend(children)

        return parents
