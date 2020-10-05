import open3d as o3d
import numpy as np
import copy
import math

###### 関数設定 ######

# 2つの点群を描画する
def draw_registration_result(source, target, transformation = np.identity(4)):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])



# データの読み込みと加工
class prepare_dataset:

    def __init__(self, pc_pass):
        self.pcd = o3d.io.read_point_cloud(pc_pass)
        self.pcd_down = copy.deepcopy(self.pcd)

    def downsampling(self, voxel_size):
        print(":: Downsample with a voxel size %.3f." % voxel_size)
        self.pcd_down = self.pcd.voxel_down_sample(voxel_size)

    def estimate_normal(self, radius_normal):
        if self.pcd_down.has_normals() == True:
            print(":: Already have normal")
        else:
            print(":: Estimate normal with search radius %.3f." % radius_normal)
            self.pcd_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    def calculate_fpfh(self, radius_feature):
        print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        self.pcd_fpfh = o3d.registration.compute_fpfh_feature(
            self.pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))



# FPFH and RANSACでRegistrationを行う
def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, 
        target_down, 
        source_fpfh, 
        target_fpfh, 
        distance_threshold, # max_correspondence_distance
        o3d.registration.TransformationEstimationPointToPoint(False), # estimation_method
        4, # ransac_n, 点何個ランダムサンプリングするのか
        [o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)], # チェックの方法,リスト何個か指定可能
        o3d.registration.RANSACConvergenceCriteria(4000000, 500)) # 収束判定, max_iteration=100000, and max_validation=100
    return result
