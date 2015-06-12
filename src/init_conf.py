'''
generate different initialization and submit to job Queue
'''
import input_gen
from /Users/Bo/Projects/PyScheduler/src import addjob
import numpy as np
from src import *

if __name__=="__main__":

    username='bz16@nyu.edu'
    funcname='baseline.py'

    lrate_base=0.01
    lrate_rates=2.0
    learningrates=lrate_base*(lrate_rates**np.arange(100))

    inps=inputs_gen()
    num_of_ins=len(inps)
    comments=[]
    email_flags=[]
    for i in num_of_ins:
        comments.append('lrate-eta:{0}'.format(i))
        email_flags.append(str((i+1)/(i+1)))
    comments=tuple(comments)
    addjobs(username,funcname,inps,comments,email_flags)
    
