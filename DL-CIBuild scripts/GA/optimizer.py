"""
Class that holds a genetic algorithm for evolving a population of params.
"""
from functools import reduce
from operator import add
import random
from GA.solution import Solution
"""Class that implements genetic algorithm for Hyper-parameter tuning"""
class Optimizer():
    
    def __init__(self, GA_params, all_possible_params):
        """Create an optimizer."""
        self.random_select = GA_params["random_select"]
        self.mutate_chance = GA_params["mutate_chance"]
        self.retain = GA_params["retain"]
        self.all_possible_params = all_possible_params
    
    def create_population(self, count):
        """Create a population of random solutions."""
        pop = []
        for _ in range(0, count):
            # Create a random solution.
            solution = Solution(self.all_possible_params)
            solution.create_random()
            # Add the solution to our population.
            pop.append(solution)
        return pop

    @staticmethod
    def fitness(solution):
        """Return the score, which is our fitness function."""
        return solution.score

    def grade(self, pop):
        """Find average fitness for a population. """
        summed = reduce(add, (self.fitness(solution) for solution in pop))
        return summed / float((len(pop)))

    def crossover(self, mother, father):
        """Make two children as parts of their parents.
        Args:
            mother (dict): parameters
            father (dict): parameters
        Returns:
            (list): combined params
        """
        children = []
        for _ in range(2):
            child = {}
            # Loop through the parameters and pick params for the kid.
            for param in self.all_possible_params:
                child[param] = random.choice([mother.params[param], father.params[param]] )

            solution = Solution(self.all_possible_params)
            solution.set_params(child)
            # Randomly mutate some of the children.
            if self.mutate_chance > random.random():
                solution = self.mutate(solution)
            children.append(solution)
        return children
    
    
    def mutate(self, solution):
        """Randomly mutate one part of the solution."""
        # Choose a random key.
        mutation = random.choice(list(self.all_possible_params.keys()))
        # Mutate one of the params.
        solution.params[mutation] = random.choice(self.all_possible_params[mutation])
        return solution
    
    """Evolve a population of solutions."""
    def evolve(self, pop):
        #Get scores for each solution.
        graded = [(self.fitness(solution), solution) for solution in pop]
        #"Sort on the scores.
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]
        #Get the number we want to keep for the next gen.
        retain_length = int(len(graded)*self.retain)
        # define what we want to keep.
        parents = graded[:retain_length]
        # For those we aren't keeping, randomly keep some anyway.
        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)
        # Now find out how many spots we have left to fill.
        parents_length = len(parents)
        desired_length = len(pop) - parents_length
        
        # Add children, which are bred from two remaining solutions.
        if parents_length > 1 and desired_length> 0:
            children = []
            while len(children) < desired_length:
                if parents_length==2:
                    male_index = 1
                    female_index = 0
                else:
                    male_index = random.randint(0, parents_length-1)
                    female_index = random.randint(0, parents_length-1)
                
                # Assuming they aren't the same solutions...
                if male_index != female_index:
                    print("Get a random mom and dad.")
                    male = parents[male_index]
                    female = parents[female_index]
                    # crossover them.
                    babies = self.crossover(male, female)
                    # Add the children one at a time.
                    for baby in babies:
                        # Don't grow larger than desired length.
                        if len(children) < desired_length:
                            children.append(baby)
            parents.extend(children)
        return parents