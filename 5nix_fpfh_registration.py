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

# --- 推定したリガンドのポーズと真のポーズとのＲＳＭＥをファイルに書き込み
f = open('result_rmse.txt', mode="w")

for i in range(20):
    # --- ＦＰＦＨ特徴量をマッチングさせる
    result = pp.execute_global_registration(source.pcd_down, 
                                            target.pcd_down, 
                                            source.pcd_fpfh, 
                                            target.pcd_fpfh, 
                                            voxel_size)

    # --- 結果表示
    print(":: Iteration %d --- --- ---" % i)
    print(":: ", result)

    # --- 画像保存（色替え、向き変え）
    target.pcd.paint_uniform_color([0, 0.651, 0.929])
    source.pcd.paint_uniform_color([1, 0.706, 0])
    source.pcd.transform(result.transformation)

    # --- 推定したリガンドのポーズと真のポーズとのＲＳＭＥ
    rmse = np.average(source.pcd.compute_point_cloud_distance(source_model.pcd))
    f.write(str(rmse)+"\n")

    # --- 画像保存（ウィンドウ出さずに保存）
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(source.pcd)
    vis.add_geometry(target.pcd)
    # - 結果画像の向きを変える試み
    # ctr = vis.get_view_control()
    # ctr.translate(90.0, 0.0)
    # -
    vis.update_geometry()
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image("result/test_pic%d.jpg" % i)
    vis.destroy_window()

    # --- 向きを元に戻す
    source.pcd.transform(np.linalg.inv(result.transformation))

f.close()