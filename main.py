from network.network import SOM
import matplotlib.pyplot as plt
import numpy as np


def f(x, y):
    return x*y

def plot_network(som):
    weights = som.network.nodes

    x = weights[:, 0]
    y = weights[:, 1]
    z = weights[:, 2]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=0.5)
    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    num_data_points = 1000
    iterations = 5
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    x = np.linspace(-1, 1, num_data_points)
    y = np.linspace(-1, 1, num_data_points)
    z = np.array(list(map(f, x, y)))
    som_input = np.transpose(np.vstack([x, y, z]))
    som_input = np.vstack([som_input for _ in range(iterations)])
    som = SOM(data_dim=3, num_row=10, num_col=10, random_interval=(-1,1), lr_step=num_data_points)
    som.process_input(data_points=som_input, random_shuffle=True)
    plot_network(som)


    # 1 learn prototypes in w^c, clustering the input space using (SOM, ITM or GNG)
    # 2 learn corresponding output vectors w_c^out
    # 3 adapt corresponding linear map A_c



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
