import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    if x > 0:
        return scale * x
    else:
        return scale * alpha * (math.exp(x) - 1.)

if __name__ == '__main__':
    
    np.random.seed(sum(map(ord, "distributions")))
    xl = list(np.linspace(-10, 10, 201))
    yl = [selu(x) for x in xl]

    use_seaborn = True

    if use_seaborn:
        sns.set(style="ticks")
        # sns.set_style('whitegrid')
        sns.despine(left=True)
        plt.plot(xl, yl)
        plt.show()
        # sns.plot(xl, yl)
        # sns.show()
    else:
        plt.plot(xl, yl)
        plt.grid(b=True, which='both', color='0.65',linestyle='--')
        # plt.grid(b=True, which='major', color='b', linestyle='-')
        # plt.grid(b=True, which='minor', color='r', linestyle='--')
        plt.show()