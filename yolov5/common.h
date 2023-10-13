#ifndef COMMON_H
#define COMMON_H

#include <torch/torch.h>
using namespace torch;
using namespace torch::nn;

// kernel, padding, dilation
int autopad(int k, int p = -1, int d = 1)
{
  // Pad to 'same' shape outputs
  if (d > 1)
    k = d * (k - 1) + 1;
  if (p == -1)
    p = static_cast<int>(k / 2);
  return p;
}

struct Conv : Module
{
  Conv(int c1, int c2, int k = 1, int s = 1, int p = -1, int g = 1, int d = 1, bool act = true)
  {
    conv = register_module("conv", torch::nn::Conv2d(Conv2dOptions(c1, c2, k)
                                                         .stride(s)
                                                         .padding(autopad(k, p, d))
                                                         .bias(false)
                                                         .groups(g)
                                                         .dilation(d)));

    bn = register_module("bn", BatchNorm2d(BatchNorm2dOptions(c2)));

    default_act = register_module("act", SiLU());
    // default_act = SiLU();
  }

  torch::Tensor forward(torch::Tensor x)
  {
    return default_act(bn(conv(x)));
  }

  Conv2d conv{nullptr};
  BatchNorm2d bn{nullptr};
  SiLU default_act{nullptr};
  int i;         // attach index
  int f;         // 'from' index
  std::string t; // type, default class name
  int np;        // number params
};

struct Bottleneck : Module
{
  // ch_in, ch_out, shortcut, groups, expansion
  Bottleneck(int c1, int c2, bool shortcut = true, int g = 1, float e = 0.5)
  {
    int c_ = static_cast<int>(c2 * e);
    cv1 = ModuleHolder<Conv>(c1, c_, 1, 1);
    cv2 = ModuleHolder<Conv>(c_, c2, 3, 1, -1, g);

    register_module("cv1", cv1);
    register_module("cv2", cv2);
    add = shortcut && (c1 == c2);
  }

  torch::Tensor forward(torch::Tensor x)
  {
    if (add)
    {
      return x + cv2(cv1(x));
    }
    else
    {
      return cv2(cv1(x));
    }
  }

  ModuleHolder<Conv> cv1{nullptr}, cv2{nullptr};
  bool add = false;
};

struct C3 : Module
{
  // CSP Bottleneck with 3 convolutions
  C3(int c1, int c2, int n = 1, bool shortcut = true, int g = 1, float e = 0.5) // ch_in, ch_out, number, shortcut, groups, expansion
  {
    int c_ = static_cast<int>(c2 * e);
    cv1 = ModuleHolder<Conv>(c1, c_, 1, 1);
    cv2 = ModuleHolder<Conv>(c1, c_, 1, 1);
    cv3 = ModuleHolder<Conv>(2 * c_, c2, 1);
    for (size_t i = 0; i < n; i++)
    {
      m->push_back(ModuleHolder<Bottleneck>(c_, c_, shortcut, g, 1.0));
    }
    register_module("cv1", cv1);
    register_module("cv2", cv2);
    register_module("cv3", cv3);
    register_module("m", m);
  }

  torch::Tensor forward(torch::Tensor x)
  {
    return cv3(torch::cat({m->forward(cv1(x)), cv2(x)}, 1));
  }

  ModuleHolder<Conv> cv1{nullptr}, cv2{nullptr}, cv3{nullptr};
  Sequential m;
  int i;         // attach index
  int f;         // 'from' index
  std::string t; // type, default class name
  int np;        // number params
};

struct SPPF : Module
{
  SPPF(int c1, int c2, int k = 5)
  {
    int c_ = static_cast<int>(c1 / 2);
    cv1 = ModuleHolder<Conv>(c1, c_, 1, 1);
    cv2 = ModuleHolder<Conv>(c_ * 4, c2, 1, 1);
    m = MaxPool2d(MaxPool2dOptions(k).stride(1).padding(static_cast<int>(k / 2)));

    register_module("cv1", cv1);
    register_module("cv2", cv2);
    register_module("m", m);
  }

  torch::Tensor forward(torch::Tensor x)
  {
    auto x_ = cv1(x);
    auto y1 = m(x_);
    auto y2 = m(y1);
    return cv2(torch::cat({x_, y1, y2, m(y2)}, 1));
  }

  ModuleHolder<Conv> cv1{nullptr}, cv2{nullptr};
  MaxPool2d m{nullptr};
  int i;         // attach index
  int f;         // 'from' index
  std::string t; // type, default class name
  int np;        // number params
};

struct Concat : Module
{
  Concat(int dimension = 1)
  {
    d = dimension;
  }

  torch::Tensor forward(std::vector<Tensor> x)
  {
    // for (size_t i = 0; i < f.size(); i++)
    // {
    //   std::cout << f[i] << " ";
    // }
    // std::cout << std::endl;
    // for (size_t i = 0; i < x.size(); i++)
    // {
    //   std::cout << x[i].sizes() << std::endl;
    // }

    // auto x_ = torch::cat(x, d);
    return torch::cat(x, d);
  }

  int d;
  int i;              // attach index
  std::vector<int> f; // 'from' index
  std::string t;      // type, default class name
  int np;             // number params
};

struct Proto : Module
{
  Proto(int c1, int c_ = 256, int c2 = 32)
  {
    cv1 = ModuleHolder<Conv>(c1, c_, 3);
    upsample = Upsample(UpsampleOptions()
                            .scale_factor(std::vector<double>({double(2), double(2)}))
                            .mode(torch::kNearest));
    cv2 = ModuleHolder<Conv>(c_, c_, 3);
    cv3 = ModuleHolder<Conv>(c_, c2);

    register_module("cv1", cv1);
    register_module("upsample", upsample);
    register_module("cv2", cv2);
    register_module("cv3", cv3);
  }

  torch::Tensor forward(torch::Tensor x)
  {
    return cv3(cv2(upsample(cv1(x))));
  }

  ModuleHolder<Conv> cv1{nullptr}, cv2{nullptr}, cv3{nullptr};
  Upsample upsample{nullptr};
};
#endif