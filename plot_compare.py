import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    #load in the training data input range
    x = np.load('x.npy')
    y = np.load('y.npy')
    #load in the true test data
    x_s = np.load('x_s.npy')
    y_test = np.load('y_test.npy')
    #load in the regression data from the GP model
    y_gp = np.load('y_gp.npy')
    mu = np.load('mu_s.npy')
    std = np.load('std.npy')
    #load in the regression data from the deep learning model
    y_nn = np.load('y_nn.npy')

    #define the unseen range
    unseen_range = np.array([x[-1], x_s[-1]]).ravel()
    ylim = np.array([np.min(y_test) * 2.5, np.max(y_test) * 1.30])
    #now lets plot it all
    plt.figure()
    plt.plot(x_s, y_gp, 'r', label='Posterior')
    plt.plot(x_s, y_test, 'b', label='True')
    plt.xlim([x_s[0], x_s[-1]])
    plt.title('Gaussian Process')
    plt.gca().fill_between(unseen_range,
                           ylim[0],
                           ylim[1],
                           color="#f3c4eb")
    plt.gca().fill_between(x_s.flat,
                           mu.ravel() - 2 * std,
                           mu.ravel() + 2 * std,
                           color="#dddddd")
    plt.scatter(x.ravel(),y.ravel(),label='Training Samples')
    plt.legend()
    plt.ylim(ylim)
    plt.savefig('gp.eps', format='eps')
    plt.show()
    plt.close()



    plt.figure()
    plt.plot(x_s, y_nn.ravel(), 'r', label='Fitted')
    plt.plot(x_s, y_test, 'b', label='True')
    plt.xlim([x_s[0], x_s[-1]])
    plt.title('Deep Neural Network')
    plt.gca().fill_between(unseen_range,
                           ylim[0],
                           ylim[1],
                           color="#f3c4eb")
    plt.scatter(x.ravel(),y.ravel(),label='Training Samples')
    plt.legend()
    plt.ylim(ylim)
    plt.savefig('nn.eps', format='eps')
    plt.show()
    plt.close()


    
