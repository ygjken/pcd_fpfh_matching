import os 
import numpy as np 


rmse = np.array([["" for j in range(20)] for i in range(5, 51, 5)])


i = 0 
for fpfh_radius_size in range(5, 51, 5):
    with  open('result/result_rmse/result_rmse_radius%d.txt' % fpfh_radius_size, mode="r") as f:
        rmse[i] = np.array([s.strip() for s in f.readlines()], dtype='float64')
    i = i + 1


print(rmse)
