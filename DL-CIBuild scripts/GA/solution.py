"""Class that represents the solution to be evolved."""
import random
class Solution():
    def __init__(self, all_possible_params):
        self.entry = {}
        self.score = 0.
        self.all_possible_params = all_possible_params
        self.params = {}  #  represents model parameters to be picked by creat_random method
        self.model = None
        
    """Create the model random params."""
    def create_random(self):
        for key in self.all_possible_params:
            self.params[key] = random.choice(self.all_possible_params[key])

    def set_params(self, params):
        self.params = params
      
    """
        Train the model and record the score.
    """
    def train_model(self, fn_train,params_fn):
        
        if self.score == 0.:
                res = fn_train(self.params,params_fn)
                self.score =  res["entry"]["F1"] #1-float(res["validation_loss"])
                self.model = res["model"]
                self.entry = res['entry']
            
    """Print out a network."""
    def print_solution(self):
        print("for params ", self.params , "the score in the train = ",self.score)