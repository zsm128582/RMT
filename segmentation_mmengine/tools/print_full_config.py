from mmengine.config import Config

# 加载你的叶子节点配置文件
cfg = Config.fromfile('/home/u2023110769/code/RMT/segmentation_mmengine/configs/RMT_T/RMT_S_upernet_8xb2-160k_ade20k_512x512.py')

# 打印展开后的完整配置
print(cfg.pretty_text)

# 如果你想保存到文件以便查看
cfg.dump('full_config.py')