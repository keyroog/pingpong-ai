import numpy as np
import math

class Discretizer:
    """
    Classe per discretizzare stati continui in discreti.
    """
    def __init__(self, bins_per_dimension):
        """
        :param bins_per_dimension: Lista con il numero di intervalli per ogni dimensione dello stato.
        """
        self.bins_per_dimension = bins_per_dimension

    def discretize(self, continuous_state):
        """
        Discretizza uno stato continuo.
        :param continuous_state: Array numpy con lo stato continuo.
        :return: Tupla con lo stato discretizzato.
        """
        discrete_state = []
        for value, bins in zip(continuous_state, self.bins_per_dimension):
            discrete_value = math.floor(value * bins)
            discrete_state.append(discrete_value)
        return tuple(discrete_state)

    def get_state_space_size(self):
        """
        Calcola la dimensione dello spazio discreto totale.
        :return: Lista delle dimensioni dello spazio discreto per ogni variabile.
        """
        return [bins for bins in self.bins_per_dimension]