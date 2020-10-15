import open3d as o3d
import numpy as np
import copy
import math



def draw_registration_result(source, target, transformation = np.identity(4)):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])



def save_img(source, target, file_path, file_name, transformation = np.identity(4)):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(source.transform(transformation))
    vis.add_geometry(target)
    vis.update_geometry()
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(file_path + "/" + file_name + ".jpg")
    vis.destroy_window()



def change_pcd_color(source, target):
    source.paint_uniform_color([1, 0.706, 0])
    target.paint_uniform_color([0, 0.651, 0.929])



class prepare_dataset:

    def __init__(self, file_pass):
        self.pcd = o3d.io.read_point_cloud(file_pass)
        self.pcd_viewdata = copy.deepcopy(self.pcd)
    
    def downsampling(self, voxel_size):
        self.pcd = self.pcd.voxel_down_sample(voxel_size)
    
    def estimate_normal(self, radius, max_nn):
        if self.pcd.has_normals() == True:
            print(":: Already have normal")
        else:
            print(":: Estimate normal with search radius %.3f." % radius)
            print("::                      max_nn  %d." % max_nn)
            self.pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    
    def calculate_fpfh(self, radius, max_nn):
        print(":: Compute FPFH feature with search radius %.3f." % radius)
        print("::                           max_nn  %d." % max_nn)
        self.pcd_fpfh = o3d.registration.compute_fpfh_feature(
            self.pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100))



def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, 
                                distance_threshold, transformation_type, n_ransac, 
                                similarity_threshold, max_iter, max_valid):

    FUNCTIONS = {'PointToPlane': o3d.registration.TransformationEstimationPointToPlane,
    'PointToPoint': o3d.registration.TransformationEstimationPointToPoint}

    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, 
        target_down, 
        source_fpfh, 
        target_fpfh, 
        max_correspondence_distance=distance_threshold,
        estimation_method=FUNCTIONS[transformation_type](), 
        ransac_n=n_ransac,
        checkers=[o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(similarity_threshold),
        o3d.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        criteria=o3d.registration.RANSACConvergenceCriteria(max_iter, max_valid)) 
    return result

