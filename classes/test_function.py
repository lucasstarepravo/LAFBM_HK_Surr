from functions.test_function import test_function, dif_analytical, laplace_phi, dif_do


class TestFunction:
    def __init__(self, coordinates, weights):
        self.surface_value = test_function(coordinates)
        self.dtdx_true = dif_analytical(coordinates, 'dtdx')
        self.dtdy_true = dif_analytical(coordinates, 'dtdy')
        self.laplace_true = laplace_phi(coordinates)
        self.dtdx_approx = dif_do(weights, self.surface_value, 'dtdx')
        self.dtdy_approx = dif_do(weights, self.surface_value, 'dtdy')
        self.laplace_approx = dif_do(weights, self.surface_value, 'Laplace')
