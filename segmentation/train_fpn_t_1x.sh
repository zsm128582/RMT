# sudo pip uninstall mmseg
# pip install openmim
# mim install mmengine
# pip uninstall mmcv
# pip install mmcv-full
# pip install cython==0.29.33
# pip uninstall pycocotools
# pip install mmpycocotools
# sudo python3 setup.py install
# pip install terminaltables
# pip install efficientnet_pytorch
# pip install importlib_metadata 
bash tools/dist_train.sh configs/RMT/RMT_FPN_t_1x.py 2 --options model.pretrained=/home/zengshimao/code/RMT/segmentation/rmt_t_70epoch.pth