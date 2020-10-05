import pointclouds_preprocess as pp
import open3d as o3d 
import numpy as np
import copy


target = pp.prepare_dataset("../../data/5nix_compound/5nix_pocket_pc.ply")
source = pp.prepare_dataset("../../data/5nix_compound/5nix_ligand_pc.ply")
source_model = pp.prepare_dataset("../../data/5nix_compound/model_ligand.ply")


voxel_size = 0.5


source.downsampling(voxel_size)
target.downsampling(voxel_size)
source.estimate_normal(voxel_size * 2)
target.estimate_normal(voxel_size * 2)
source.calculate_fpfh(voxel_size * 30)
target.calculate_fpfh(voxel_size * 30)


result = pp.execute_global_registration(source.pcd_down, 
                                            target.pcd_down, 
                                            source.pcd_fpfh, 
                                            target.pcd_fpfh, 
                                            voxel_size)


print(":: ", result)

# 変換前
target.pcd.paint_uniform_color([0, 0.651, 0.929])
source.pcd.paint_uniform_color([1, 0.706, 0])
pp.draw_registration_result(source.pcd, target.pcd, np.identity(4))

# 変換後
source.pcd.transform(result.transformation)
pp.draw_registration_result(source.pcd, target.pcd, np.identity(4))

print(":: 正解ポーズとの平均自乗誤差 -> ", np.average(source.pcd.compute_point_cloud_distance(source_model.pcd)))

# 元に戻す
source.pcd.transform(np.linalg.inv(result.transformation))
pp.draw_registration_result(source.pcd, target.pcd, np.identity(4))




"""
print(":: 推定した回転行列 ---")
print(result.transformation)
print(":: 推定した回転行列の逆行列 ---")
print(np.linalg.inv(result.transformation))

print(np.dot(result.transformation, np.linalg.inv(result.transformation)))
"""