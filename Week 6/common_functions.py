import matplotlib.pyplot as plt
import tensorflow as tf

def sparse_categorical_crossentropy(target, predictions):
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target, logits=predictions
    )
    return tf.reduce_mean(losses)


def plot_results(iteration_history, loss_history):
    plt.plot(iteration_history, loss_history)
    plt.xlabel("Iter")
    plt.ylabel("Loss")
    plt.title("Loss vs Iterations")
    plt.show()


# Taken from Tensorflow Documentation
class Adam:
    def __init__(self, learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.t = 1.0
        self.v_dvar, self.s_dvar = [], []
        self.built = False

    def apply_gradients(self, grads, vars):
        if not self.built:
            for var in vars:
                v = tf.Variable(tf.zeros(shape=var.shape))
                s = tf.Variable(tf.zeros(shape=var.shape))
                self.v_dvar.append(v)
                self.s_dvar.append(s)
            self.built = True
        for i, (grad, var) in enumerate(zip(grads, vars)):
            self.v_dvar[i].assign(
                self.beta_1 * self.v_dvar[i] + (1 - self.beta_1) * grad
            )
            self.s_dvar[i].assign(
                self.beta_2 * self.s_dvar[i] + (1 - self.beta_2) * tf.square(grad)
            )
            v_dvar_bc = self.v_dvar[i] / (1 - (self.beta_1**self.t))
            s_dvar_bc = self.s_dvar[i] / (1 - (self.beta_2**self.t))
            var.assign_sub(
                self.learning_rate * (v_dvar_bc / (tf.sqrt(s_dvar_bc) + self.epsilon))
            )
        self.t += 1.0
        return
