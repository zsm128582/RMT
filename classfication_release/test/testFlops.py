import torch
import torch.nn as nn
from thop import profile

from Restormer import TransformerBlock
from RMT import RetBlock
from RMT import RetNetRelPos2d

import time

### test Restormer Attention

# model = TransformerBlock(dim= 64 , num_heads=4 , ffn_expansion_factor= 3 , bias= True , LayerNorm_type="With")
# input = torch.randn(1,  64  , 224  ,224)
# flops , params = profile(model , inputs=(input,))


## test RMT block

model = RetBlock(retention="whole", embed_dim=64, num_heads=4, ffn_dim=3 * 64)
relpos = RetNetRelPos2d(embed_dim=64, num_heads=4, initial_value=1e-5, heads_range=4)
pos = relpos((224, 224), chunkwise_recurrent=False)
input = torch.randn(64, 224, 224, 64)
start_time = time.time()

flops, params = profile(model, inputs=(input, None, False, pos))
end_time = time.time()

print(f"Time for section 1: {end_time - start_time} seconds")
print(f"flops:{flops}")
