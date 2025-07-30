import matplotlib.pyplot as plt
import arviz as az

class ResultsInterpreter:
    def __init__(self, trace, series):
        self.trace = trace
        self.series = series

    def plot_trace(self):
        az.plot_trace(self.trace)
        plt.show()

    def plot_change_point(self):
        tau_samples = self.trace.posterior['tau'].values.flatten()
        plt.hist(tau_samples, bins=30)
        plt.title('Posterior of Change Point (tau)')
        plt.xlabel('Index')
        plt.ylabel('Frequency')
        plt.show()