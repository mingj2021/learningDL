#ifndef UNET_MODEL_H
#define UNET_MODEL_H
#include "unet_parts.h"

struct UNet : Module
{
    UNet(int n_channels, int n_classes, bool bilinear = false) : m_n_channels(n_channels), m_n_classes(n_classes), m_bilinear(bilinear)
    {
        inc = ModuleHolder<DoubleConv>(n_channels, 64);
        register_module("inc", inc);
        down1 = ModuleHolder<Down>(64, 128);
        register_module("down1", down1);
        down2 = ModuleHolder<Down>(128, 256);
        register_module("down2", down2);
        down3 = ModuleHolder<Down>(256, 512);
        register_module("down3", down3);
        int factor = 2;
        if (bilinear)
            factor = 2;
        else
            factor = 1;
        down4 = ModuleHolder<Down>(512, int(1024 / factor));
        register_module("down4", down4);
        up1 = ModuleHolder<Up>(1024, int(512 / factor), bilinear);
        register_module("up1", up1);
        up2 = ModuleHolder<Up>(512, int(256 / factor), bilinear);
        register_module("up2", up2);
        up3 = ModuleHolder<Up>(256, int(128 / factor), bilinear);
        register_module("up3", up3);
        up4 = ModuleHolder<Up>(128, 64, bilinear);
        register_module("up4", up4);
        outc = ModuleHolder<OutConv>(64, n_classes);
        register_module("outc", outc);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        auto x1 = inc(x);
        auto x2 = down1(x1);
        auto x3 = down2(x2);
        auto x4 = down3(x3);
        auto x5 = down4(x4);
        x = up1(x5, x4);
        x = up2(x, x3);
        x = up3(x, x2);
        x = up4(x, x1);
        auto logits = outc(x);
        return logits;
    }
    int m_n_channels, m_n_classes;
    bool m_bilinear;

    ModuleHolder<DoubleConv> inc{nullptr};
    ModuleHolder<Down> down1{nullptr}, down2{nullptr}, down3{nullptr}, down4{nullptr};
    ModuleHolder<Up> up1{nullptr}, up2{nullptr}, up3{nullptr}, up4{nullptr};
    ModuleHolder<OutConv> outc{nullptr};
};

#endif