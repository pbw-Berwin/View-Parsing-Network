# View-Parsing-Network
main model for cross-view semantic segmentation

##Training Command
CUDA_VISIBLE_DEVICES=0 python train.py --fc-dim 256 --use-depth false --use-mask true --transform-type fc --input-resolution 400 --label-res 25 --store-name [Your storename] --n-views 8 --batch-size 2 -j 10
