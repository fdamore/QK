import matplotlib
import matplotlib.pyplot as plt
import numpy as np



#calback function
class QKCallback:
    
    num_iteration = 0
   

    def __init__(self) -> None:
          self._data = [[] for i in range(5)]

    #callback function  
    def callback(self, x0, x1=None, x2=None, x3=None, x4=None):
            
            """
            Args:
                x0: number of function evaluations
                x1: the parameters
                x2: the function value
                x3: the stepsize
                x4: whether the step was accepted
            """

            self.num_iteration +=1

            print(f'**********************')
            print(f'Print callback. Iteration {self.num_iteration}')
            print(f'Number of function evaluations: {x0}')
            print(f'The paramenters: {x1}')
            print(f'The function value: {x2}')
            print(f'The stepsize: {x3}')
            print(f'Whether the step was accepted: {x4}')
            print(f'**********************')

            self._data[0].append(x0)
            self._data[1].append(x1)
            self._data[2].append(x2)
            self._data[3].append(x3)
            self._data[4].append(x4)

            #return True if you want stop the training
            stop_training = False
            return stop_training
    
    def plot_data(self):
          plt.rcParams["font.size"] = 20
          #fig, ax = plt.subplots(1, 2, figsize=(14, 5))#plt.subplots(1, 2, figsize=(14, 5))
          plt.plot([i + 1 for i in range(len(self._data[0]))], np.array(self._data[2]), c="k", marker="o")
          #ax[0].set_xlabel("Iterations")
          #ax[0].set_ylabel("Loss")
          #fig.tight_layout()
          plt.show()
    

