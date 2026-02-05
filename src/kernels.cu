#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
template <typename T>
__global__ void trace_kernel(const T* input, T* output, size_t rows, size_t cols){
  size_t Max = min(rows, cols); // 计算下算几个对角线的数
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;// 计算下这是第几个线程
  size_t jump_number = blockDim.x * gridDim.x;// 一个线程每次循环跳的数，便于后面内存合并优化

  T sum = 0.0;
  for(size_t i = idx; i < Max; i += jump_number)
    sum += input[i * cols + i];//加对角线上的数

  atomicAdd(output, sum);//将各个线程的数加起来求和
}
template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  // TODO: Implement the trace function
 
  // 设置变量
  T *input, *output;

  // 分配显存，为输入输出开辟空间
  cudaMalloc(&input, rows * cols * sizeof(T));
  cudaMalloc(&output, sizeof(T));

  // 拷贝数据
  cudaMemcpy(input, h_input.data(), rows * cols * sizeof(T), cudaMemcpyHostToDevice);

  // 初始化结果
  cudaMemset(output, 0, sizeof(T));

  // 核函数计算
  trace_kernel<T><<<256,256>>>(input, output, rows, cols);
 
  //拷贝结果
  T answer = 0;
  cudaMemcpy(&answer, output, sizeof(T), cudaMemcpyDeviceToHost);

  //没用的显存free掉
  cudaFree(input);
  cudaFree(output);

  // 返回结果
  return answer;
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 *
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
__global__ void flashAttention_kernel(const T* h_q,      
    const T* h_k,      
    const T* h_v,      
    T* o,            
    int batch_size,  
    int target_seq_len,
    int src_seq_len,
    int query_heads,
    int kv_heads,
    int head_dim,
    bool is_causal){
      
      // 准备变量
      int batch_idx = blockIdx.z;
      int head_idx  = blockIdx.y;
      int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
      int tid = threadIdx.x;
      int kv_head_idx = head_idx / (query_heads / kv_heads);
      size_t q_offset = ((size_t)batch_idx * target_seq_len * query_heads +
                       (size_t)query_idx * query_heads + head_idx) * head_dim;
      float scale = 1.0f/sqrtf((float)head_dim);
      float sum = 0.0f, Max = -1e38f;
      float arr[256];
      #pragma unroll
      for(int t = 0; t < head_dim; ++t)
        arr[t] = 0.0f;
      
      bool valid_q = (query_idx < target_seq_len);// 先前的越界逻辑

      // 引入共享内存和分块优化
      
      // 搬运数据
      float q[256];
      if (valid_q) {
        #pragma unroll
        for(int i =0; i < head_dim; ++ i){
          q[i] = (float) h_q[q_offset + i];
        }
      }
      const int tile_count = 16;
      __shared__ float k[tile_count][260];
      __shared__ float v[tile_count][260];

      for(int i = 0; i < src_seq_len; i += tile_count){
        bool use_vectorization = std::is_same<T, float>::value && (head_dim % 4 == 0);
        // 当数据类型为float时引入向量化搬运
        if(use_vectorization){
          // 强转指针
          const float4* k_ptr = reinterpret_cast<const float4 *>(h_k);
          const float4* v_ptr = reinterpret_cast<const float4 *>(h_v);
          // 一次搬四个，要除4
          int vec_dim = head_dim / 4;
          
          //其余逻辑与后面差不多，就是引入了向量化
          #pragma unroll// blockDim固定，可以展开
          for(int j = tid; j < tile_count * vec_dim; j += blockDim.x){
            int row = j / vec_dim;
            int col = j % vec_dim;
            int kv_idx = i + row;

            size_t kv_base = ((size_t)batch_idx * src_seq_len * kv_heads + 
                                    (size_t)kv_idx * kv_heads + kv_head_idx) *  vec_dim ;
            int col_base = col * 4;                        

            if(kv_idx < src_seq_len){
              // 读取数据
              float4 k_val = k_ptr[kv_base + col];
              float4 v_val = v_ptr[kv_base + col];
              
              // 拆包
              k[row][col_base + 0] = k_val.x;
              k[row][col_base + 1] = k_val.y;
              k[row][col_base + 2] = k_val.z;
              k[row][col_base + 3] = k_val.w;

              v[row][col_base + 0] = v_val.x;
              v[row][col_base + 1] = v_val.y;
              v[row][col_base + 2] = v_val.z;
              v[row][col_base + 3] = v_val.w;
            }
            // 内存开多了就初始化为0
            else{
              k[row][col_base + 0] = 0.0f;
              k[row][col_base + 1] = 0.0f;
              k[row][col_base + 2] = 0.0f;
              k[row][col_base + 3] = 0.0f;

              v[row][col_base + 0] = 0.0f;
              v[row][col_base + 1] = 0.0f;
              v[row][col_base + 2] = 0.0f;
              v[row][col_base + 3] = 0.0f;
            }
          }
        }
        else{
          #pragma unroll 
          for(int j = tid; j < tile_count * head_dim; j += blockDim.x){
            int row = j / head_dim;
            int col = j % head_dim;
            int kv_idx = i + row;

            size_t kv_base = ((size_t)batch_idx * src_seq_len * kv_heads + 
                                    (size_t)kv_idx * kv_heads + kv_head_idx) * head_dim;
            
            if(kv_idx < src_seq_len){
              k[row][col] = (float)h_k[kv_base + col];
              v[row][col] = (float)h_v[kv_base + col];
            }
            // 内存开多了就初始化为0
            else{
              k[row][col] = 0.0f;
              v[row][col] = 0.0f;
            }
          }
            
        }
        // 等所有人搬完数据再继续
        __syncthreads();

        if (valid_q) {
          #pragma unroll 
          for(int t = 0; t < tile_count;++ t){

            // 特判跳过条件
            int current_idx = i + t;
            if(current_idx >= src_seq_len)break;
            if(is_causal && current_idx > query_idx)continue;

            float now_sum = 0.0f;

            // online softmax 优化朴素softmax算法，核心是将softmax的多个流程优化为一个流程，边计算边更改

            // 维护 online softmax 所需变量
            #pragma unroll
            for(int d = 0; d < head_dim; ++d)
              now_sum += (float)q[d] * (float)k[t][d];
            now_sum *= scale;

            float Max_pre = Max;
            Max = fmaxf(Max_pre, now_sum);

            float add = expf(now_sum - Max);
            float change = expf(Max_pre - Max);

            // 公式1：sum_new = sum_old * change + add
            sum = sum * change + add;

            // 公式2: O_new = O_old * change + V_curr * add
            #pragma unroll
            for(int d = 0; d < head_dim; ++ d){
              arr[d] = arr[d] * change + v[t][d] * add;
            }
          }
        }
        __syncthreads();
      }
      if (valid_q) {
        // 防止除以 0
        if (sum == 0.0f) sum = 1.0f;

        // 将结果写回o数组  
        T* o_ptr = o + q_offset;
        for(int i = 0; i < head_dim; ++ i){
          o_ptr[i] = (T)(arr[i]/sum);
        }
      }
}
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {      
  // TODO: Implement the flash attention function
  // 设置变量
  size_t q_size = static_cast<size_t>(batch_size) * target_seq_len * query_heads * head_dim;
  size_t k_size = static_cast<size_t>(batch_size) * src_seq_len * kv_heads * head_dim;
  size_t v_size = k_size;
  size_t o_size = q_size;

  T *q, *k, *v, *o;

  // 分配显存，为输入输出开辟空间
  cudaMalloc(&q, q_size * sizeof(T));
  cudaMalloc(&k, k_size * sizeof(T));
  cudaMalloc(&v, v_size * sizeof(T));
  cudaMalloc(&o, o_size * sizeof(T));

  // 拷贝数据
  cudaMemcpy(q, h_q.data(), q_size * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(k, h_k.data(), k_size * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(v, h_v.data(), v_size * sizeof(T), cudaMemcpyHostToDevice);

  // 初始化结果
  cudaMemset(o, 0, o_size * sizeof(T));

  // 核函数计算
  dim3 block(128);
  dim3 grid((target_seq_len + 127) / 128, query_heads, batch_size);
  flashAttention_kernel<T><<<grid, block>>>(q, k, v, o,
                                          batch_size, target_seq_len, src_seq_len,
                                          query_heads, kv_heads, head_dim,
                                          is_causal);
 
  //拷贝结果
  cudaMemcpy(h_o.data(), o, o_size * sizeof(T), cudaMemcpyDeviceToHost);

  //没用的显存free掉
  cudaFree(q);
  cudaFree(k);
  cudaFree(v);
  cudaFree(o);

}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);

