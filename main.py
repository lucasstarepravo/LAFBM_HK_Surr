from classes.simulation import run
from functions.plot import plot_convergence
from functions.plot import plot_convergence2
from functions.plot import plot_convergence3
from functions.plot import plot_convergence4
import dill as pickle


if __name__ == '__main__':
    total_nodes_list = [10, 20, 50, 100, 200, 400]
    polynomial_list = [2, 2, 2, 2, 2, 2]
    results = run(total_nodes_list, polynomial_list)


#with open('results.dill', 'wb') as f:
#    pickle.dump(results, f)
#plot_convergence(results, 'dtdx')
#plot_convergence(results, 'dtdy')
plot_convergence4(results, 'laplace')

#plot_convergence2(results, 'dtdx')
#plot_convergence2(results, 'dtdy')
#plot_convergence2(results, 'Laplace')