"""Class that represents the network to be evolved."""
import random
import logging
from train import train_model

class Network():
    def __init__(self, nn_param_choices=None):
        """Initialize our network.
        Args:  nn_param_choices (dict): Parameters for the network
        """
        self.accuracy = 0.
        self.completeScore = {}
        self.nn_param_choices = nn_param_choices
        self.network = {}  # (dic): represents MLP network parameters

    def create_random(self):
        """Create a random network."""
        for key in self.nn_param_choices:
            self.network[key] = random.choice(self.nn_param_choices[key])

    def create_set(self, network):
        """Set network properties.
        Args:  network (dict): The network parameters
        """
        self.network = network
    def train(self,file_name ,train_set):
        """Train the network and record the accuracy.
        """
        if self.accuracy == 0.:
            self.completeScore = train_model(self.network, file_name ,train_set)
            print(self.completeScore)
            self.accuracy = self.completeScore["AUC"]

    def print_network(self):
        """Print out a network."""
        logging.info(self.network)
        logging.info(self.completeScore)