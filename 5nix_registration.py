import preprocess as pp
import open3d as o3d 
import numpy as np
import copy
import os


target = pp.prepare_dataset("../../data/5nix_compound/5nix_pocket_pc.ply")
source = pp.prepare_dataset("../../data/5nix_compound/5nix_ligand_pc.ply")
source_model = pp.prepare_dataset("../../data/5nix_compound/model_ligand.ply")
file_path = "result_usePapersParameters"
f = open(file_path + '/rmse.txt', mode="w")


pp.change_pcd_color(source.pcd, target.pcd)

source.estimate_normal(3.1, 471)
target.estimate_normal(3.1, 471)

source.calculate_fpfh(3.1, 135)
target.calculate_fpfh(3.1, 135)


for i in range(30):
    result = pp.execute_global_registration(source.pcd, target.pcd,
                                            source.pcd_fpfh, target.pcd_fpfh,
                                            1.5, 'PointToPoint', 4, 
                                            0.9, 4000000, 500)
    
    source.pcd.transform(result.transformation)

    rmse = np.average(source.pcd.compute_point_cloud_distance(source_model.pcd))
    f.write(str(rmse)+"\n") 

    pp.save_img(source.pcd, target.pcd, file_path, "result_pose%d" % i)

    source.pcd.transform(np.linalg.inv(result.transformation))

f.close()

