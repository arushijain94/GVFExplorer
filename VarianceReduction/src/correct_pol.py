import os
import numpy as np

loc = '/home/mila/j/jainarus/scratch/VarReduction/tmaze/ideal_targets'
for i in range(1,5):
    pol = np.load(os.path.join(loc, 'pol_'+str(i)+ '.npy'))
    pol = pol/np.linalg.norm(pol, ord=1, keepdims=True, axis=1)
    pol[:,-1] = 1.0 - np.sum(pol[:, :-1], axis=1)
    print("pol1:", pol)
    np.save(os.path.join(loc, 'pol_'+str(i)+ '.npy'), pol)
