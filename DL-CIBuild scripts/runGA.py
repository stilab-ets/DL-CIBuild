import logging
from optimizer import Optimizer
from tqdm import tqdm
# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename = 'LOGGING GA.txt'
)
def train_networks(networks,file_name,train_set):
    """Train each network.
    Args:
        networks (list): Current population of networks
        dataset (str): Dataset to use for training/evaluating
    """
    pbar = tqdm(total=len(networks))
    for network in networks:
        network.train(file_name,train_set)
        pbar.update(1)
    pbar.close()

def get_average_accuracy(networks):
    """Get the average accuracy for a group of networks.
    Args:
        networks (list): List of networks
    Returns:
        float: The average accuracy of a population of networks.
    """
    total_accuracy = 0
    for network in networks:
        total_accuracy += network.accuracy

    return total_accuracy / len(networks)

def generate(generations, population, nn_param_choices,file_name,train_set):
    logging.info("***Evolving %d generations with population %d***" %(generations, population))
    """Generate a network with the genetic algorithm.
    Args:
        generations (int): Number of times to evole the population
        population (int): Number of networks in each generation
        nn_param_choices (dict): Parameter choices for networks
        file_name (str): project name
        train_set : Dataset to use for training/evaluating
    """
    optimizer = Optimizer(nn_param_choices)
    networks = optimizer.create_population(population)
    # Evolve the generation.
    for i in range(generations):
        logging.info("***Doing generation %d of %d***" %   (i + 1, generations))
        # Train and get accuracy for networks.
        train_networks(networks,file_name,train_set)
        # Get the average accuracy for this generation.
        average_accuracy = get_average_accuracy(networks)
        # Print out the average accuracy each generation.
        logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
        logging.info('-'*80)
        # Evolve, except on the last iteration.
        if i != generations - 1:
            logging.info(" # Do the evolution.")
            networks = optimizer.evolve(networks)
        else:
            logging.info(" # Do not do the evolution.",i,generations - 1)
    logging.info(" # Sorting our final population.")
    networks = sorted(networks, key=lambda x: x.accuracy, reverse=True)
    logging.info(" # our final population is sorted.")
    # Print out the top 5 networks.
    print_networks(networks[:3])
    return networks[0]

def print_networks(networks):
    """Print a list of networks.
    Args:
        networks (list): The population of networks
    """
    logging.info('-'*80)
    logging.info(" Print top networks")
    for network in networks:
        network.print_network()    
