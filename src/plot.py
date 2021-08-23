import matplotlib.pyplot as plt

def plot(real, predicted):
    plt.gcf().set_size_inches(22, 15, forward=True)

    plt.plot(real, label='real')
    plt.plot(predicted, label='predicted')

    plt.legend(['Real', 'Predicted'])

    plt.show()
