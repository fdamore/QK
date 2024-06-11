#calback function
class QKCallback:
    
    num_iteration = 0;

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

            #return True if you want stop the training
            stop_training = False
            return stop_training