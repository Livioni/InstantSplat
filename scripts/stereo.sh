# # Copyright (C) 2024 Denso IT Laboratory, Inc.
# # All Rights Reserved

DATA_ROOT_DIR="assets/carla/town10"

echo "create database"

python scripts/create_db.py  -out $DATA_ROOT_DIR -r $DATA_ROOT_DIR
colmap feature_extractor --database_path $DATA_ROOT_DIR/database.db --image_path $DATA_ROOT_DIR/images --ImageReader.camera_model PINHOLE
colmap exhaustive_matcher --database_path $DATA_ROOT_DIR/database.db
mkdir -p $DATA_ROOT_DIR/sparse/0
colmap point_triangulator --database_path $DATA_ROOT_DIR/database.db --image_path $DATA_ROOT_DIR/images --input_path $DATA_ROOT_DIR/sparse --output_path $DATA_ROOT_DIR/sparse/0 --Mapper.tri_ignore_two_view_tracks=0
rm -r $DATA_ROOT_DIR/sparse/*.txt
# rm $DATA_ROOT_DIR/database.db
colmap image_undistorter \
    --image_path $DATA_ROOT_DIR/images \
    --input_path $DATA_ROOT_DIR/sparse/0 \
    --output_path $DATA_ROOT_DIR/dense \
    --output_type COLMAP

# 2. 执行 PatchMatch Stereo
#    在 $DATA_ROOT_DIR/dense 文件夹下执行多视图立体，输出深度图等文件到 densify/ 之类的子目录里
colmap patch_match_stereo \
    --workspace_path $DATA_ROOT_DIR/dense \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true

# 3. 立体融合（将深度图融合为点云）
colmap stereo_fusion \
    --workspace_path $DATA_ROOT_DIR/dense \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path $DATA_ROOT_DIR/dense/fused.ply

colmap poisson_mesher \
    --input_path $DATA_ROOT_DIR/dense/fused.ply \
    --output_path $DATA_ROOT_DIR/dense/meshed-poisson.ply

echo "Dense reconstruction done!"