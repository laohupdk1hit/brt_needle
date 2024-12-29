from sklearn.metrics.pairwise import pairwise_distances as p_d
import numpy as np

ends= np.loadtxt('E:/Program Files/Slicer 4.11.20210226/slicerscripts/deln_ne1.dat',delimiter=',')
print(ends.shape)

dist_ends = p_d(ends)
np.savetxt('dist_ends1.txt',dist_ends,fmt='%.8f')
print(dist_ends.shape)