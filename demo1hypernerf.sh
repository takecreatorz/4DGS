
set -e



#bash colmap.sh data/hypernerf/misc/americano hypernerf
#python scripts/downsample_point.py data/hypernerf/misc/americano/colmap/dense/workspace/fused.ply data/hypernerf/misc/americano/points3D_downsample2.ply
python train.py -s  data/hypernerf/misc/americano/ --port 6017 --expname "hypernerf/misc/americano" --configs arguments/hypernerf/default.py
python render.py --model_path "output/hypernerf/misc/americano/"  --skip_train --configs arguments/hypernerf/default.py
python metrics.py --model_path "output/hypernerf/misc/americano/"

bash colmap.sh data/hypernerf/misc/cross-hands1 hypernerf
python scripts/downsample_point.py data/hypernerf/misc/cross-hands1/colmap/dense/workspace/fused.ply data/hypernerf/misc/cross-hands1/points3D_downsample2.ply
python train.py -s  data/hypernerf/misc/cross-hands1/ --port 6017 --expname "hypernerf/misc/cross-hands1" --configs arguments/hypernerf/default.py
python render.py --model_path "output/hypernerf/misc/cross-hands1/"  --skip_train --configs arguments/hypernerf/default.py
python metrics.py --model_path "output/hypernerf/misc/cross-hands1/"

bash colmap.sh data/hypernerf/misc/espresso hypernerf
python scripts/downsample_point.py data/hypernerf/misc/espresso/colmap/dense/workspace/fused.ply data/hypernerf/misc/espresso/points3D_downsample2.ply
python train.py -s  data/hypernerf/misc/espresso/ --port 6017 --expname "hypernerf/misc/espresso" --configs arguments/hypernerf/default.py
python render.py --model_path "output/hypernerf/misc/espresso/"  --skip_train --configs arguments/hypernerf/default.py
python metrics.py --model_path "output/hypernerf/misc/espresso/"

bash colmap.sh data/hypernerf/misc/keyboard hypernerf
python scripts/downsample_point.py data/hypernerf/misc/keyboard/colmap/dense/workspace/fused.ply data/hypernerf/misc/keyboard/points3D_downsample2.ply
python train.py -s  data/hypernerf/misc/keyboard/ --port 6017 --expname "hypernerf/misc/keyboard" --configs arguments/hypernerf/default.py
python render.py --model_path "output/hypernerf/misc/keyboard/"  --skip_train --configs arguments/hypernerf/default.py
python metrics.py --model_path "output/hypernerf/misc/keyboard/"

bash colmap.sh data/hypernerf/misc/oven-mitts hypernerf
python scripts/downsample_point.py data/hypernerf/misc/oven-mitts/colmap/dense/workspace/fused.ply data/hypernerf/misc/oven-mitts/points3D_downsample2.ply
python train.py -s  data/hypernerf/misc/oven-mitts/ --port 6017 --expname "hypernerf/misc/oven-mitts" --configs arguments/hypernerf/default.py
python render.py --model_path "output/hypernerf/misc/oven-mitts/"  --skip_train --configs arguments/hypernerf/default.py
python metrics.py --model_path "output/hypernerf/misc/oven-mitts/"

bash colmap.sh data/hypernerf/misc/split-cookie hypernerf
python scripts/downsample_point.py data/hypernerf/misc/split-cookie/colmap/dense/workspace/fused.ply data/hypernerf/misc/split-cookie/points3D_downsample2.ply
python train.py -s  data/hypernerf/misc/split-cookie/ --port 6017 --expname "hypernerf/misc/split-cookie" --configs arguments/hypernerf/default.py
python render.py --model_path "output/hypernerf/misc/split-cookie/"  --skip_train --configs arguments/hypernerf/default.py
python metrics.py --model_path "output/hypernerf/misc/split-cookie/"

bash colmap.sh data/hypernerf/misc/tamping hypernerf
python scripts/downsample_point.py data/hypernerf/misc/tamping/colmap/dense/workspace/fused.ply data/hypernerf/misc/tamping/points3D_downsample2.ply
python train.py -s  data/hypernerf/misc/tamping/ --port 6017 --expname "hypernerf/misc/tamping" --configs arguments/hypernerf/default.py
python render.py --model_path "output/hypernerf/misc/tamping/"  --skip_train --configs arguments/hypernerf/default.py
python metrics.py --model_path "output/hypernerf/misc/tamping/"