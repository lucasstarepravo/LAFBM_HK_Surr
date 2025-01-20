from functions.discrete_operator import calc_weights


class Weights:
    def __init__(self, coordinates, polynomial, h, total_nodes, s):
        self.x, self.y, self.laplace, self._neigh_coor = calc_weights(coordinates, polynomial, h, total_nodes, s)
