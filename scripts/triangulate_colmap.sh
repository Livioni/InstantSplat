# Copyright (C) 2024 Denso IT Laboratory, Inc.
# All Rights Reserved

DATA_ROOT_DIR="assets/carla/town10"

echo "create database"

python scripts/create_db.py  -out $DATA_ROOT_DIR -r $DATA_ROOT_DIR
colmap feature_extractor --database_path $DATA_ROOT_DIR/database.db --image_path $DATA_ROOT_DIR/images --ImageReader.camera_model PINHOLE
colmap exhaustive_matcher --database_path $DATA_ROOT_DIR/database.db
mkdir -p $DATA_ROOT_DIR/sparse/0 
colmap point_triangulator --database_path $DATA_ROOT_DIR/database.db --image_path $DATA_ROOT_DIR/images --input_path $DATA_ROOT_DIR/sparse --output_path $DATA_ROOT_DIR/sparse/0 --Mapper.tri_ignore_two_view_tracks=0
rm -r $DATA_ROOT_DIR/sparse/*.txt
rm $DATA_ROOT_DIR/database.db
