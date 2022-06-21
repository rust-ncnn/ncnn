// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "multiheadattention.h"

#include <float.h>
#ifdef NCNN_INT8
#include <math.h>
#endif

namespace ncnn {

MultiHeadAttention::MultiHeadAttention()
{
}

int MultiHeadAttention::load_param(const ParamDict& pd)
{
    embed_dim = pd.get(0, 0);
    num_head = pd.get(1, 1);
    weight_data_size = pd.get(2, 0);
    int8_scale_term = pd.get(3, 0);

    if (int8_scale_term)
    {
#if NCNN_INT8
        support_int8_storage = true;
#else
        NCNN_LOGE("please build ncnn with NCNN_INT8 enabled for int8 inference");
        return -1;
#endif
    }
    return 0;
}

int MultiHeadAttention::load_model(const ModelBin& mb)
{
#define LOAD_MAT(name, len) \
    name = mb.load(len, 0); \
    if (name.empty())       \
    {                       \
        return -100;        \
    }

#define LOAD_FLOAT_MAT(name, len) \
    name = mb.load(len, 1);       \
    if (name.empty())             \
    {                             \
        return -100;              \
    }

    LOAD_MAT(q_weight_data, weight_data_size);
    LOAD_FLOAT_MAT(q_bias_data, embed_dim);

    LOAD_MAT(k_weight_data, weight_data_size);
    LOAD_FLOAT_MAT(k_bias_data, embed_dim);

    LOAD_MAT(v_weight_data, weight_data_size);
    LOAD_FLOAT_MAT(v_bias_data, embed_dim);

    LOAD_MAT(out_weight_data, weight_data_size);
    LOAD_FLOAT_MAT(out_bias_data, embed_dim);

#if NCNN_INT8
    if (int8_scale_term)
    {
        LOAD_FLOAT_MAT(q_input_scale, 1);
        LOAD_FLOAT_MAT(k_input_scale, 1);
        LOAD_FLOAT_MAT(v_input_scale, 1);

        LOAD_FLOAT_MAT(q_weight_scales, weight_data_size);
        LOAD_FLOAT_MAT(k_weight_scales, weight_data_size);
        LOAD_FLOAT_MAT(v_weight_scales, weight_data_size);
        LOAD_FLOAT_MAT(o_weight_scales, weight_data_size);

        LOAD_FLOAT_MAT(internal_scales, 5);
    }
#endif // NCNN_INT8

#undef LOAD_MAT
#undef LOAD_FLOAT_MAT

    return 0;
}

#ifdef NCNN_INT8
/**
 * @brief
 *  q_input_int8 * q_weight --> q_out_int32
 *  q_out_int32 / input_scale / weight_scale + bias --> q_out_fp32
 *  q_out_fp32 --> q_internal_int8
 * @param input
 * @param internal
 * @param input_scale
 * @param weight_scales
 * @param transpose_out
 * @return int
 */
int MultiHeadAttention::transform_input(
    const Mat& input, const Mat& weight, const Mat& bias, Mat& out_int8,
    const Mat& input_scale, const Mat& weight_scales, const float transform_scale,
    const Option& opt, bool transpose) const
{
    const int seqlen = input.h;
    const int embed_dim_per_head = embed_dim / num_head;
    const float scale = 1.0 / input_scale[0];

    Mat input_int8;
    if (input.elemsize != 1)
    {
        quantize_to_int8(input, input_int8, input_scale, opt);
    }

    Mat buffer(embed_dim_per_head, seqlen, num_head, 4u, opt.workspace_allocator);

    for (int q = 0; q < num_head; q++)
    {
        // xq = affine(q)
        if (transpose)
        {
            Mat outm = buffer.channel(q);

            for (int i = 0; i < embed_dim_per_head; i++)
            {
                for (int j = 0; j < seqlen; j++)
                {
                    const int8_t* ptr = input_int8.row<int8_t>(j);
                    const int8_t* kptr = (int8_t*)(weight.data) + embed_dim * (q * embed_dim_per_head + i);

                    int32_t sum = 0;
                    const int32_t index = q * embed_dim_per_head + j;
                    for (int k = 0; k < embed_dim; k++)
                    {
                        sum += *ptr++ * *kptr++;
                    }

                    float* outptr = outm.row(i);
                    outptr[j] = (float)sum * scale / weight_scales[index] + bias[index];
                }
            }
        }
        else
        {
            Mat outm = buffer.channel(q);

            for (int i = 0; i < seqlen; i++)
            {
                float* outptr = outm.row(i);

                for (int j = 0; j < embed_dim_per_head; j++)
                {
                    const int8_t* ptr = input_int8.row<int8_t>(i);
                    const int8_t* kptr = (int8_t*)(weight.data) + embed_dim * (q * embed_dim_per_head + j);

                    int32_t sum = 0;
                    const int32_t index = q * embed_dim_per_head + j;
                    for (int k = 0; k < embed_dim; k++)
                    {
                        sum += *ptr++ * *kptr++;
                    }

                    outptr[j] = (float)sum * scale / weight_scales[index] + bias[index];
                }
            }
        }
    }

    Mat transform(1, 4u, opt.workspace_allocator);
    transform[0] = transform_scale;
    quantize_to_int8(buffer, out_int8, transform, opt);
    return 0;
}

static inline int32_t int_polynominal(const int32_t x, const float s)
{
    // ax**2 + bx + c
    const float coef0 = 0.35815147;
    const float coef1 = 0.96963238 / coef0;
    const float coef2 = 1.0 / coef0;

    const int32_t b_int = floor(coef1 * s);
    const int32_t c_int = floor(coef2 * s * s);
    return x * (x + b_int) + c_int;
}

static inline int64_t int_exp(int32_t x, float s)
{
#define LN2 (-0.6931f)
    const int n = 30;
    const int x0_int = floor(LN2 / s);

    x = std::max(x, n * x0_int);
    const int q = floor(x * 1.0f / x0_int);
    const int r = x - x0_int * q;
    int64_t exp_int = int_polynominal(r, 1.0f / s);

    exp_int = std::max((int64_t)0, (int64_t)floor(exp_int * pow(2, (n - q))));
    // fprintf(stdout, "[x,r,exp_int   %d\t,%d\t,%ld\t]\n", x, r, exp_int);
    return exp_int;
#undef LN2
}

static inline float fast_pow2_multiply_3(const int32_t index)
{
    //   assert(index <= 30 && index >= -1);
    static float table[] = {
        1.5f, 3, 6, 12, 24, 48,
        96, 192, 384, 768, 1536, 3072,
        6144, 12288, 24576, 49152, 98304, 196608,
        393216, 786432, 1572864, 3145728, 6291456, 12582912,
        25165824, 50331648, 100663296, 201326592, 402653184, 805306368,
        1610612736, 3221225472
    };
    return table[index + 1];
}

static inline int32_t find_first_one(int32_t v)
{
    int pos = 0;

    if (v > 0xffff)
    {
        v >>= 16;
        pos += 16;
    }

    if (v > 0xff)
    {
        v >>= 8;
        pos += 8;
    }

    if (v > 0xf)
    {
        v >>= 4;
        pos += 4;
    }

    if (v > 0x3)
    {
        v >>= 2;
        pos += 2;
    }

    if (v > 0x1)
    {
        v >>= 1;
        pos += 1;
    }

    return pos;
}

static inline int32_t clamp(const float val, const int32_t min, const int32_t max)
{
    return std::min(max, std::max((int32_t)(val + 0.5f), min));
}

int MultiHeadAttention::log_int_softmax(int8_t* ptr, int64_t* buffer, int8_t* out, const int len, float scale) const
{
    int32_t max = ptr[0];
    for (int i = 0; i < len; ++i)
    {
        if (max < ptr[i])
        {
            max = ptr[i];
        }
    }

    int64_t sum = 0;
    for (int i = 0; i < len; ++i)
    {
        ptr[i] = ptr[i] - max;
        buffer[i] = int_exp(ptr[i], scale);
        sum += buffer[i];
    }

    for (int i = 0; i < len; ++i)
    {
        ptr[i] = int32_t(sum * 1.f / buffer[i] + 0.5f);
    }

    // log_round
    const int UINT4_MAX = 15;
    for (int i = 0; i < len; ++i)
    {
        // int32_t val = floor(log2(ptr[i]));
        int32_t val = find_first_one(ptr[i]);
        float big = fast_pow2_multiply_3(val - 1);
        // fprintf(stdout, "val %d\n", val);
        if (ptr[i] >= big)
        {
            // fprintf(stdout, "big index %d\n", i);
            val += 1;
        }

        if (val > UINT4_MAX)
        {
            out[i] = -1;
            continue;
        }

        out[i] = UINT4_MAX - val;
    }

    return 0;
}

/**
 * @brief int8 mha, referenced to
 *
 *  https://github.com/megvii-research/FQ-ViT/blob/main/models/vit_quant.py#L95
 *
 * @param bottom_blobs
 * @param top_blobs
 * @param opt
 * @return int
 */
int MultiHeadAttention::forward_int8(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    // mha int8 kernel
    const Mat& q_blob = bottom_blobs[0];
    const Mat& k_blob = bottom_blobs.size() == 1 ? q_blob : bottom_blobs[1];
    const Mat& v_blob = bottom_blobs.size() == 1 ? q_blob : bottom_blobs[2];

    // 145
    const int seqlen = q_blob.h;
    // 64
    const int embed_dim_per_head = embed_dim / num_head;

    Option opt_g = opt;
    opt_g.blob_allocator = opt.workspace_allocator;
    opt_g.use_packing_layout = false;

    // 64, 145, 12
    Mat xq(embed_dim_per_head, seqlen, num_head, 1u, opt.workspace_allocator);
    Mat xk(embed_dim_per_head, seqlen, num_head, 1u, opt.workspace_allocator);
    // transpose(v) for better gemm performance
    Mat xv(seqlen, embed_dim_per_head, num_head, 1u, opt.workspace_allocator);

    transform_input(q_blob, q_weight_data, q_bias_data, xq, q_input_scale, q_weight_scales, internal_scales[0], opt_g);
    transform_input(k_blob, k_weight_data, k_bias_data, xk, k_input_scale, k_weight_scales, internal_scales[1], opt_g);
    transform_input(v_blob, v_weight_data, v_bias_data, xv, v_input_scale, v_weight_scales, internal_scales[2], opt_g, true);

    // xq @ qk * inv_sqrt_embed_dim_per_head
    const float inv_sqrt_embed_dim_per_head = 1.f / sqrt(embed_dim_per_head);
    Mat xqk(seqlen, seqlen, num_head, 1u, opt.workspace_allocator);
    // xqk = xq * xk
    // xq  (embed_dim_per_head, seqlen)
    // xk  (embed_dim_per_head, seqlen)
    {
        float out_scale = inv_sqrt_embed_dim_per_head * internal_scales[3] / (internal_scales[0] * internal_scales[1]);

        for (int q = 0; q < num_head; ++q)
        {
            const Mat xqm = xq.channel(q);
            const Mat xkm = xk.channel(q);

            Mat outm = xqk.channel(q);

            for (int i = 0; i < seqlen; i++)
            {
                int8_t* outptr = outm.row<int8_t>(i);

                for (int j = 0; j < seqlen; j++)
                {
                    const int8_t* qptr = xqm.row<int8_t>(i);
                    const int8_t* kptr = xkm.row<int8_t>(j);

                    int32_t sum = 0.f;
                    for (int k = 0; k < embed_dim_per_head; k++)
                    {
                        sum += *qptr++ * *kptr++;
                    }

                    outptr[j] = clamp(sum * out_scale, -127, 127);
                }
            }
        }
    }

    // int_softmax(xqk)
    Mat xqk_uint4(seqlen, seqlen, num_head, 1u, opt.workspace_allocator);
    {
        Mat buffer(seqlen, 8u, opt.workspace_allocator);
        int64_t* bufptr = (int64_t*)buffer.data;
        float attn_scale = internal_scales[3];
        // TODO int_softmax
        for (int q = 0; q < num_head; ++q)
        {
            Mat inm = xqk.channel(q);
            Mat outm = xqk_uint4.channel(q);

            for (int i = 0; i < seqlen; ++i)
            {
                int8_t* inptr = inm.row<int8_t>(i);
                int8_t* outptr = outm.row<int8_t>(i);
                log_int_softmax(inptr, bufptr, outptr, seqlen, attn_scale);
            }
        }
    }

    // xqkv int4 @ int8, implement by shift
    Mat xqkv(embed_dim_per_head, num_head, seqlen, 4u, opt.workspace_allocator);
    for (int q = 0; q < num_head; ++q)
    {
        // xqkv = xqk * xv
        // xqk (seqlen, seqlen)
        // xv  (seqlen, embed_dim_per_head)
        // out (embed_dim_per_head, num_head, seqlen)
        const Mat xqkm = xqk_uint4.channel(q);
        const Mat xvm = xv.channel(q);

        for (int i = 0; i < seqlen; i++)
        {
            int32_t* outptr = xqkv.channel(i).row<int32_t>(q);

            for (int j = 0; j < embed_dim_per_head; j++)
            {
                const int8_t* qkptr = xqkm.row<int8_t>(i);
                const int8_t* vptr = xvm.row<int8_t>(j);

                int32_t sum = 0;
                for (int k = 0; k < seqlen; k++)
                {
                    if (*vptr == 0)
                    {
                        continue;
                    }
                    sum += (*qkptr++) << (*vptr++);
                }

                outptr[j] = sum;
            }
        }
    }

    Mat& top_blob = top_blobs[0];
    top_blob.create(embed_dim, seqlen, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -1;
    // out = affine(xqkv)
    // xqkv  (embed_dim, seqlen)
    for (int i = 0; i < seqlen; i++)
    {
        float* outptr = top_blob.row(i);

        for (int j = 0; j < embed_dim; j++)
        {
            const int32_t* ptr = xqkv.channel(i);
            const int8_t* kptr = (const int8_t*)out_weight_data + embed_dim * j;

            float out_scale = 1.0 / o_weight_scales[j];
            int32_t sum = 0;
            for (int k = 0; k < embed_dim; k++)
            {
                sum += *ptr++ * *kptr++;
            }

            outptr[j] = (sum >> 15) * out_scale + out_bias_data[j];
        }
    }

    return 0;
}
#endif

// refers to https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
int MultiHeadAttention::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
#if NCNN_INT8
    if (opt.use_int8_inference)
    {
        if (q_weight_data.elemsize != (size_t)1u || k_weight_data.elemsize != (size_t)1u || v_weight_data.elemsize != (size_t)1u || out_weight_data.elemsize != (size_t)1u)
        {
            return -1;
        }
        return forward_int8(bottom_blobs, top_blobs, opt);
    }
#endif
    const Mat& q_blob = bottom_blobs[0];
    const Mat& k_blob = bottom_blobs.size() == 1 ? q_blob : bottom_blobs[1];
    const Mat& v_blob = bottom_blobs.size() == 1 ? q_blob : bottom_blobs[2];

    const int seqlen = q_blob.h;
    const int embed_dim_per_head = embed_dim / num_head;

    Mat& top_blob = top_blobs[0];
    top_blob.create(embed_dim, seqlen, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -1;

    Mat xq(embed_dim_per_head, seqlen, num_head, 4u, opt.workspace_allocator);
    Mat xk(embed_dim_per_head, seqlen, num_head, 4u, opt.workspace_allocator);
    Mat xv(seqlen, embed_dim_per_head, num_head, 4u, opt.workspace_allocator);

    Mat xqk(seqlen, seqlen, num_head, 4u, opt.workspace_allocator);

    Mat xqkv(embed_dim_per_head, num_head, seqlen, 4u, opt.workspace_allocator);

    const float inv_sqrt_embed_dim_per_head = 1.f / sqrt(embed_dim_per_head);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < num_head; q++)
    {
        // xq = affine(q) * inv_sqrt_embed_dim_per_head
        {
            Mat outm = xq.channel(q);

            for (int i = 0; i < seqlen; i++)
            {
                float* outptr = outm.row(i);

                for (int j = 0; j < embed_dim_per_head; j++)
                {
                    const float* ptr = q_blob.row(i);
                    const float* kptr = (const float*)q_weight_data + embed_dim * (q * embed_dim_per_head + j);

                    float sum = q_bias_data[q * embed_dim_per_head + j];
                    for (int k = 0; k < embed_dim; k++)
                    {
                        sum += *ptr++ * *kptr++;
                    }

                    outptr[j] = sum * inv_sqrt_embed_dim_per_head;
                }
            }
        }

        // xk = affine(k)
        {
            Mat outm = xk.channel(q);

            for (int i = 0; i < seqlen; i++)
            {
                float* outptr = outm.row(i);

                for (int j = 0; j < embed_dim_per_head; j++)
                {
                    const float* ptr = k_blob.row(i);
                    const float* kptr = (const float*)k_weight_data + embed_dim * (q * embed_dim_per_head + j);

                    float sum = k_bias_data[q * embed_dim_per_head + j];
                    for (int k = 0; k < embed_dim; k++)
                    {
                        sum += *ptr++ * *kptr++;
                    }

                    outptr[j] = sum;
                }
            }
        }

        // xv = affine(v)
        {
            Mat outm = xv.channel(q);

            for (int i = 0; i < embed_dim_per_head; i++)
            {
                for (int j = 0; j < seqlen; j++)
                {
                    const float* ptr = v_blob.row(j);
                    const float* kptr = (const float*)v_weight_data + embed_dim * (q * embed_dim_per_head + i);

                    float sum = v_bias_data[q * embed_dim_per_head + i];
                    for (int k = 0; k < embed_dim; k++)
                    {
                        sum += *ptr++ * *kptr++;
                    }

                    float* outptr = outm.row(i);

                    outptr[j] = sum;
                }
            }
        }

        // xqk = xq * xk
        // xq  (embed_dim_per_head, seqlen)
        // xk  (embed_dim_per_head, seqlen)
        {
            const Mat xqm = xq.channel(q);
            const Mat xkm = xk.channel(q);

            Mat outm = xqk.channel(q);

            for (int i = 0; i < seqlen; i++)
            {
                float* outptr = outm.row(i);

                for (int j = 0; j < seqlen; j++)
                {
                    const float* qptr = xqm.row(i);
                    const float* kptr = xkm.row(j);

                    float sum = 0.f;
                    for (int k = 0; k < embed_dim_per_head; k++)
                    {
                        sum += *qptr++ * *kptr++;
                    }

                    outptr[j] = sum;
                }
            }
        }

        // softmax(xqk)
        {
            Mat outm = xqk.channel(q);

            for (int i = 0; i < seqlen; i++)
            {
                float* ptr = outm.row(i);

                float max = -FLT_MAX;
                for (int j = 0; j < seqlen; j++)
                {
                    max = std::max(max, ptr[j]);
                }

                float sum = 0.f;
                for (int j = 0; j < seqlen; j++)
                {
                    ptr[j] = (float)(exp(ptr[j] - max));
                    sum += ptr[j];
                }

                for (int j = 0; j < seqlen; j++)
                {
                    ptr[j] /= sum;
                }
            }
        }

        // xqkv = xqk * xv
        // xqk (seqlen, seqlen)
        // xv  (seqlen, embed_dim_per_head)
        // out (embed_dim_per_head, num_head, seqlen)
        {
            const Mat xqkm = xqk.channel(q);
            const Mat xvm = xv.channel(q);

            for (int i = 0; i < seqlen; i++)
            {
                float* outptr = xqkv.channel(i).row(q);

                for (int j = 0; j < embed_dim_per_head; j++)
                {
                    const float* qkptr = xqkm.row(i);
                    const float* vptr = xvm.row(j);

                    float sum = 0.f;
                    for (int k = 0; k < seqlen; k++)
                    {
                        sum += *qkptr++ * *vptr++;
                    }

                    outptr[j] = sum;
                }
            }
        }
    }

    // out = affine(xqkv)
    // xqkv  (embed_dim, seqlen)
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < seqlen; i++)
    {
        float* outptr = top_blob.row(i);

        for (int j = 0; j < embed_dim; j++)
        {
            const float* ptr = xqkv.channel(i);
            const float* kptr = (const float*)out_weight_data + embed_dim * j;

            float sum = out_bias_data[j];
            for (int k = 0; k < embed_dim; k++)
            {
                sum += *ptr++ * *kptr++;
            }

            outptr[j] = sum;
        }
    }

    return 0;
}

} // namespace ncnn
