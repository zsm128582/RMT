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
# pip install prettytable
bash tools/dist_train.sh configs/RMT/RMT_Uper_s_2x.py 8 --options model.pretrained=/home/zengshimao/code/RMT/segmentation/RMT-S.pth