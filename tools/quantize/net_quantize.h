#pragma once
// ncnn private header
#include "../modelwriter.h"

class NetQuantize : public ModelWriter
{
public:
    NetQuantize()
    {
    }
    // conv and gemm quant param
    std::map<std::string, ncnn::Mat> blob_int8scale_table;
    std::map<std::string, ncnn::Mat> weight_int8scale_table;

    // MutiHeadAttention quant param
    std::map<std::string, std::shared_ptr<ini::Table> > mha_table;

public:
    bool read_txt_format(const char* path);
    bool read_ini_format(const char* path);

    int quantize_convolution();
    int quantize_convolutiondepthwise();
    int quantize_innerproduct();
    int quantize_mha();
    int fuse_requantize();

    void set_weight_suffix(std::string s);

    ncnn::Mat convert();

private:
    std::string suffix;
};
