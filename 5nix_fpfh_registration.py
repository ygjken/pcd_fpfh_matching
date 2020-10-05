import pointclouds_preprocess as pp
import open3d as o3d 
import numpy as np
import copy


target = pp.prepare_dataset("../../data/5nix_compound/5nix_pocket_pc.ply")
source = pp.prepare_dataset("../../data/5nix_compound/5nix_ligand_pc.ply")


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


pp.draw_registration_result(source.pcd, target.pcd, result.transformation)




target_pcd = copy.deepcopy(target.pcd.paint_uniform_color([0, 0.651, 0.929]))
source_pcd = copy.deepcopy(source.pcd.transform(result.transformation))
source_pcd = source_pcd.paint_uniform_color([1, 0.706, 0])

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(source_pcd)
vis.add_geometry(target_pcd)
vis.update_geometry()
vis.poll_events()
vis.update_renderer()
vis.capture_screen_image("result/test_pic.jpg")
vis.destroy_window()

