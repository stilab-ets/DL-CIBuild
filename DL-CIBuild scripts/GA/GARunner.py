from GA.optimizer import Optimizer
from tqdm import tqdm
import threading
import Utils
def train_sol_thread(solution,fn_train,params_fn,i):
    solution.train_model(fn_train,params_fn)
    print("solution ", i," trained")
    
def train_population(pop, fn_train,params_fn):
    pbar = tqdm(total=len(pop))
    threads = list()
    i=1
    for solution in pop:
        x = threading.Thread(target=train_sol_thread, args=(solution,fn_train,params_fn,i))
        i=i+1
        threads.append(x)
        x.start()
        pbar.update(1)
        
    for index, thread in enumerate(threads):
        thread.join()
    pbar.close()


def get_average_score(pop):
    """Get the average score for a group of solutions."""
    total_scores = 0
    for solution in pop:
        total_scores += solution.score
    return total_scores / len(pop)

"""Generate the optimal params with the genetic algorithm."""
""" Args:
        GA_params: Params for GA
        all_possible_params (dict): Parameter choices for the model
        train_set : training dataset
        fn_train : a function used to compute the prediction accuracy
"""
def generate(all_possible_params, fn_train , params_fn):
   
    GA_params = {
            "population_size": Utils.nbr_sol,
            "max_generations": Utils.nbr_gen,
            "retain": 0.7,
            "random_select":0.1,
            "mutate_chance":0.1
            }
    
    print("params of GA" , GA_params)
    optimizer = Optimizer(GA_params ,all_possible_params)
    pop = optimizer.create_population(GA_params['population_size'])
    # Evolve the generation.
    for i in range(GA_params['max_generations']):
        print("*********************************** REP(GA) ",(i+1))
        # Train and get accuracy for solutions.
        train_population(pop,fn_train,params_fn)
        # Get the average accuracy for this generation.
        average_accuracy = get_average_score(pop)
        # Print out the average accuracy each generation.
        print("Generation average: %.2f%%" % (average_accuracy * 100))
        # Evolve, except on the last iteration.
        if i != (GA_params['max_generations']):
            print("Generation evolving..")
            evolved = optimizer.evolve(pop)
            if(len(evolved)!=0):
                pop=evolved
        else:
            pop = sorted(pop, key=lambda x: x.score, reverse=True)
    # Print out the top 2 solutions.
    size = len(pop)
    if size < 3:
        print_pop(pop[:size])
    else:
        print_pop(pop[:3])
    return pop[0].params ,pop[0].model,pop[0].entry

def print_pop(pop):
    for solution in pop:
        solution.print_solution()    
