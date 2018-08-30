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
import collections

def merge_two_dicts(x, y):
	"""Helper method.
			
	Given two dicts, merge them into a new dict as a shallow copy.
			
	"""
	z = x.copy()
	z.update(y)
	return z

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
        child1 = {}
        child2 = {}
                
        ordered_nn_param_choices = collections.OrderedDict(self.nn_param_choices)
                
        pos = int(0.5*len(mother))
        for i in range(len(ordered_nn_param_choices)):
            param = list(ordered_nn_param_choices.items())[i][0]
            if i < pos:
                child1[param] = mother[param]
                child2[param] = father[param]
            else:
                child1[param] = father[param]
                child2[param] = mother[param]
                
        # Now create network objects.
        child1n = Network(self.nn_param_choices)
        child1n.create_set(child1)
                
        child2n = Network(self.nn_param_choices)
        child2n.create_set(child2)

        # Randomly mutate some of the children.
        if self.mutate_chance > random.random():
            child1n = self.mutate(child1n)
        if self.mutate_chance > random.random():
            child2n = self.mutate(child2n)

        children.append(child1n)
        children.append(child2n)

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
        elif network.network[selected_layers] > 0:
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
                male_index = random.randint(0, cutoff)
            else:
                male_index = random.randint(cutoff+1, len(graded)-1)
                
            if random.random() <= 0.75:
                female_index = random.randint(0, cutoff)
            else:
                female_index = random.randint(cutoff+1, len(graded)-1)

            # Assuming they aren't the same network...
            if male_index != female_index:
                male = graded[male_index].get_parameters()
                female = graded[female_index].get_parameters()

                # Breed them.
                babies = self.breed(male, female)

                # Add the children one at a time.
                for baby in babies:
                    # Don't grow larger than desired length.
                    if len(children) < desired_length:
                        children.append(baby)

        parents.extend(children)

        return parents
