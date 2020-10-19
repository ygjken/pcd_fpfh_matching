import preprocess as pp
import open3d as o3d 
import numpy as np
import copy
import os


# prepare dataset
source_model = pp.prepare_dataset("../../data/5nix_compound/model_ligand.ply")
target_model = pp.prepare_dataset("../../data/5nix_compound/model_pocket.ply")
source = copy.deepcopy(source_model)

# change color
pp.change_pcd_color(source_model.pcd, target_model.pcd)
source.pcd.paint_uniform_color([0.91, 0.65, 0.82])

# make initial state of the pose
trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], 
                        [1.0, 0.0, 0.0, 1.0],
                        [0.0, 1.0, 0.0, 0.0], 
                        [0.0, 0.0, 0.0, 1.0]])
source.pcd.transform(trans_init)

# estimate normal
source.estimate_normal(3.1, 471)
target_model.estimate_normal(3.1, 471)

# compute fpfh feature
source.calculate_fpfh(3.1, 135)
target_model.calculate_fpfh(3.1, 135)

# folder path that store result figures
file_path = 'result_usePapersParameters'
f = open(file_path + '/rmse.txt', mode='w')


for i in range(50):
    result = pp.execute_global_registration(source.pcd, target_model.pcd,
                                            source.pcd_fpfh, target_model.pcd_fpfh,
                                            1.5, 'PointToPoint', 4, 
                                            0.9, 4000000, 500)
    
    source.pcd.transform(result.transformation)

    rmse = np.average(source.pcd.compute_point_cloud_distance(source_model.pcd))
    f.write(str(rmse)+"\n") 

    pp.save_img(source.pcd, target_model.pcd, file_path, "result_pose%d" % i)

    source.pcd.transform(np.linalg.inv(result.transformation))

f.close()

