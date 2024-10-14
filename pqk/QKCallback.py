import datetime
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import json



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
            self._data[1].append(x1.tolist())
            self._data[2].append(x2)
            self._data[3].append(x3)
            self._data[4].append(x4)

            #return True if you want stop the training
            stop_training = False
            return stop_training
    
    def plot_data(self):
        QKCallback._plot_data(self._data)

    #plot data from list of data
    def _plot_data(data):
        plt.rcParams["font.size"] = 20          
        plt.plot([i + 1 for i in range(len(data[0]))], np.array(data[2]), c="k", marker="o")
        plt.show()


    #save my feature map
    def save(self, prefix = ''):
        #create a csv file with feature maps
        current_timestamp = time.time()
        datetime_object = datetime.datetime.fromtimestamp(current_timestamp)
        formatted_datetime = datetime_object.strftime("%Y%m%d%H%M%S")
        csv_file = '../qfm/callback/' + prefix + str(formatted_datetime) + '.json'

        json_str = json.dumps(self._data, indent= 3)        

        main_path = os.path.dirname(__file__)
        file_path = os.path.join(main_path, csv_file)

        #store the features map
        with open(file_path, 'w') as f:
            f.write(json_str)

    def plot_data_file(file = ''):
        data_to_plot = None
        
        # open file and read the content in a list
        with open(file, 'r') as _file:
            content = _file.read()
            data_to_plot = json.loads(content)
                   
        QKCallback._plot_data(data_to_plot)



if __name__ == '__main__':
    ##check the reader    
    QKCallback.plot_data_file(file='qfm/callback/callback_20240704230145.json')
     
     




        
                   
                   

            
            
    

