#ifndef UNET_PARTS_H
#define UNET_PARTS_H

#include <torch/torch.h>
using torch::nn::BatchNorm2d;
using torch::nn::BatchNorm2dOptions;
using torch::nn::Conv2d;
using torch::nn::Conv2dOptions;
using torch::nn::ConvTranspose2d;
using torch::nn::ConvTranspose2dOptions;
using torch::nn::MaxPool2d;
using torch::nn::MaxPool2dOptions;
using torch::nn::ReLU;
using torch::nn::ReLUOptions;
using torch::nn::Upsample;
using torch::nn::UpsampleOptions;

using torch::nn::Module;
using torch::nn::AnyModule;
using torch::nn::ModuleHolder;
using torch::nn::Sequential;
namespace F = torch::nn::functional;

struct DoubleConv : Module
{
    DoubleConv(int in_channels, int out_channels, c10::optional<int> mid_channels=c10::nullopt)
    {
        if (mid_channels == c10::nullopt)
            mid_channels = out_channels;

        double_conv->push_back(Conv2d(Conv2dOptions(in_channels, mid_channels.value(), 3).padding(1).bias(false)));
        double_conv->push_back(BatchNorm2d(BatchNorm2dOptions(mid_channels.value())));
        double_conv->push_back(ReLU(ReLUOptions(true)));
        double_conv->push_back(Conv2d(Conv2dOptions(mid_channels.value(), out_channels, 3).padding(1).bias(false)));
        double_conv->push_back(BatchNorm2d(BatchNorm2dOptions(out_channels)));
        double_conv->push_back(ReLU(ReLUOptions(true)));
        register_module("double_conv", double_conv);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        return double_conv->forward(x);
    }
    Sequential double_conv;
};

struct Down : Module
{
    Down(int in_channels, int out_channels)
    {
        maxpool_conv->push_back(MaxPool2d(MaxPool2dOptions(2)));
        maxpool_conv->push_back(DoubleConv(in_channels, out_channels));
        register_module("maxpool_conv", maxpool_conv);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        return maxpool_conv->forward(x);
    }

    Sequential maxpool_conv;
};

struct Up : Module
{
    Up(int in_channels, int out_channels, bool bilinear = true)
    {
        if (bilinear)
        {
            up = AnyModule(Upsample(UpsampleOptions().scale_factor(std::vector<double>({2})).mode(torch::kBilinear).align_corners(true)));
            conv = ModuleHolder<DoubleConv>(in_channels, out_channels, int(in_channels / 2));
        }
        else
        {
            up = AnyModule(ConvTranspose2d(ConvTranspose2dOptions(in_channels, int(in_channels / 2), 2).stride(2)));
            conv = ModuleHolder<DoubleConv>(in_channels, out_channels);
        }
        register_module("up", up.ptr());
        register_module("conv", conv);
    }

    torch::Tensor forward(torch::Tensor x1, torch::Tensor x2)
    {
        x1 = up.forward(x1);
        auto diffY = x2.size(2) - x1.size(2);
        auto diffX = x2.size(3) - x1.size(3);
        x1 = F::pad(x1, F::PadFuncOptions({int(diffX / 2), diffX - int(diffX / 2),
                                              int(diffY / 2), diffY - int(diffY / 2)}));
        auto x = torch::cat({x2, x1}, 1);
        return conv(x);
    }
    AnyModule up;
    ModuleHolder<DoubleConv> conv{nullptr};
};

struct OutConv : Module
{
    OutConv(int in_channels, int out_channels)
    {
        conv = Conv2d(Conv2dOptions(in_channels, out_channels, 1));
        register_module("conv", conv);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        return conv(x);
    }

    Conv2d conv{nullptr};
};
#endif