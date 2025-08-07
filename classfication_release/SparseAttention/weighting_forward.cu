#include <cuda_runtime.h>
#include <cuda_fp16.h> // 可选支持半精度
#include <assert.h>
#include <torch/extension.h>
using std::min;
// 错误检查宏
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

template <typename T>
__global__ void sparse_attention_weighting_kernel(
    const T* __restrict__ atten,      // [B, H, R, W2, K, W2]
    const T* __restrict__ value,      // [B, H, R, W2, C]
    const int64_t* __restrict__ idx,      // [B, H, R, K]
    T* __restrict__ out,              // [B, H, R, W2, C]
    const int B, const int H, const int R, 
    const int W2, const int K, const int C)
{
    // 每个 block 负责一个 (b, h, r)
    int block_id = blockIdx.x;
    int b = block_id / (H * R);
    int hr = block_id % (H * R);
    int h = hr / R;
    int r = hr % R;

    // 输出 tile 的总大小为 W2 * C
    int tile_size = W2 * C;
    // 线程在 block 内的线性索引
    int stride = blockDim.x * blockDim.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

 // 每个线程负责处理多个输出位置，我们预留一个固定大小的数组（假设最大不会超过 16 个）
    const int max_local = 16;
    int local_indices[max_local];
    T local_sums[max_local];
    int count = 0;


    // 计算该线程负责的所有 tile 内的线性索引
    for (int index = tid; index < tile_size; index += stride) {
        if (count < max_local) {
            local_indices[count] = index;  // index 映射到 (i, c): i = index / C, c = index % C
            local_sums[count] = 0.0f;
            count++;
        }
    }

    // 动态共享内存，用于载入当前 key/value 区域 tile，大小为 (W2 * C) floats
    extern __shared__ double2 shared_memory[];
    T* shared_val  = reinterpret_cast<T*>(shared_memory);

     // 遍历当前 query 区域关联的 K 个 key/value 区域
    for (int j = 0; j < K; j++) {
        // 根据 idx 读取对应的 key/value 区域下标
        int idx_offset = (((b * H + h) * R + r) * K) + j;
        int region_index = idx[idx_offset];

        // 将 value[b, h, region_index, :, :]（形状 [W2, C]）载入共享内存
        for (int index = tid; index < tile_size; index += stride) {
            int l = index / C;   // 对应 value 中的像素 l
            int cc = index % C;  // 对应通道 cc
            int value_offset = ((((b * H + h) * R + region_index) * W2) + l) * C + cc;
            shared_val[index] = value[value_offset];
        }
        __syncthreads();

        // 对于每个线程负责的输出位置，计算该 j 下的加权累加
        for (int n = 0; n < count; n++) {
            int index = local_indices[n];
            int i = index / C;  // 当前 query patch 内的像素位置
            int c = index % C;  // 通道索引
            T partial = 0;
            // 累加 over l：遍历当前 key/value 区域内所有像素
            for (int l = 0; l < W2; l++) {
                // atten 的索引：[b,h,r,i,j,l]，atten 形状为 [B, H, R, W2, K, W2]
                int atten_offset = ((((b * H + h) * R + r) * W2 + i) * K + j) * W2 + l;
                T a = atten[atten_offset];
                // 从共享内存中读取对应 value 值，索引为 [l, c]
                T v = shared_val[l * C + c];
                partial += a * v;
            }
            local_sums[n] += partial;
        }
        __syncthreads();  // 同步后处理下一个 j
    }

    // 将每个线程计算的结果写入全局内存 out
    for (int n = 0; n < count; n++) {
        int index = local_indices[n];
        int i = index / C;
        int c = index % C;
        int out_offset = ((((b * H + h) * R + r) * W2 + i) * C) + c;
        out[out_offset] = local_sums[n];
    }
   
}

torch::Tensor weighting_forward(
    torch::Tensor attn,  // [B, H, R, W2, K , W2 ]
    torch::Tensor value,    // [B, H, R, W2, C]
    torch::Tensor idx,    // [B, H, R, K]
    float scale
)
{
    const int B = value.size(0);
    const int heads = value.size(1);
    const int region = value.size(2);
    const int W2 = value.size(3);
    const int C = value.size(4);
    const int K = idx.size(3);
    auto out = torch::zeros({B, heads, region, W2, C}, value.options());

    // 设置网格和块大小
    int gredDim = B*heads*region;
    dim3 threads_per_block = (16,16); // 假设每个 W2 分配 32 个线程处理 C
    

    AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "weighting_forward_cuda", ([&] {
        int shared_mem_size = (W2 * C) * sizeof(scalar_t);// atten + value 分块

        sparse_attention_weighting_kernel<scalar_t><<<gredDim, threads_per_block, shared_mem_size>>>(
            attn.data_ptr<scalar_t>(), 
            value.data_ptr<scalar_t>(), 
            idx.data_ptr<int64_t>(), 
            out.data_ptr<scalar_t>(), 
            B , heads , region , W2 , K , C
        );
    }));

    cudaDeviceSynchronize();
    return out;
}
