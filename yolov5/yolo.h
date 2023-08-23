#ifndef YOLO_H
#define YOLO_H

#include "common.h"
#include "general.h"
#include <numeric>
struct Detect : Module
{
    Detect(int nc_, std::vector<std::vector<int>> anchors_, std::vector<int> ch_, bool inplace_)
    {
        nc = nc_;
        no = nc + 5;
        nl = anchors_.size();
        na = anchors_[0].size() / 2;
        std::vector<int> v_;

        for (size_t i = 0; i < nl; i++)
        {
            v_.insert(v_.end(), anchors_[i].begin(), anchors_[i].end());
        }

        auto a = torch::from_blob(&v_[0], {int64_t(v_.size())}, torch::kInt32);
        anchors = a.to(torch::kFloat).view({nl, -1, 2}).clone();

        for (size_t i = 0; i < ch_.size(); i++)
        {
            m->push_back(torch::nn::Conv2d(Conv2dOptions(ch_[i], no * na, 1).bias(true)));
        }
        register_module("m", m);
        inplace = inplace_;
        stride = torch::tensor({8, 16, 32});
        training = true;
    }

    std::vector<Tensor> forward(std::vector<Tensor> x)
    {
        std::vector<Tensor> z;
        for (size_t i = 0; i < nl; i++)
        {
            auto module = m->ptr<Module>(i);
            auto m = module->as<Conv2dImpl>();
            int bs = x[i].size(0);
            int ny = x[i].size(2);
            int nx = x[i].size(3);
            // std::cout << " bs = " << bs << " ny = " << ny << " nx = " << nx << std::endl;
            x[i] = m->forward(x[i]);
            x[i] = x[i].view({bs, na, no, ny, nx}).permute({0, 1, 3, 4, 2}).contiguous();

            if (!training) // inference
            {
                auto [grid, anchor_grid] = _make_grid(nx, ny, i);
                // std::cout << "grid = " << grid << std::endl;
                // std::cout << "anchor_grid = " << anchor_grid << std::endl;
                auto vec = x[i].sigmoid().split({2, 2, nc + 1}, 4);
                auto xy = vec[0], wh = vec[1], conf = vec[2];
                xy = (xy * 2 + grid) * stride[i];

                wh = wh.mul(2).pow(2) * anchor_grid;
                // std::cout << wh << std::endl;
                auto y = torch::cat({xy, wh, conf}, 4);
                z.push_back(y.view({bs, na * nx * ny, no}));
                // std::cout << y.view({bs, na * nx * ny, no}).index({0,0,Slice()}) << std::endl;
            }
        }
        if (training)
        {
            return x;
        }
        else
        {
            std::vector<Tensor> o;
            o.push_back(torch::cat(z, 1));
            return o;
        }
    }

    std::tuple<Tensor, Tensor> _make_grid(int nx = 20, int ny = 20, int i = 0)
    {
        auto device = parameters()[0].device();
        anchors = anchors.to(device);
        stride = stride.to(device);
        auto d = anchors[i].options();
        // std::cout << d.device() << std::endl;
        std::vector<int64_t> shape{1, na, ny, nx, 2};
        auto y = torch::arange(ny, d).to(device);
        auto x = torch::arange(nx, d).to(device);
        std::vector<Tensor> vec = torch::meshgrid({y, x});
        Tensor yv = vec[0];
        Tensor xv = vec[1];
        auto grid = torch::stack({xv, yv}, 2).expand(shape) - 0.5; // add grid offset, i.e. y = 2.0 * x - 0.5
        auto temp = anchors[i] * stride[i];
        // std::cout << stride[i] << std::endl;
        // std::cout << anchors[i] << std::endl;
        auto anchor_grid = temp.view({1, na, 1, 1, 2}).expand(shape);
        // std::cout << "grid=" << grid.sizes() << std::endl;
        // std::cout << "anchor_grid=" << anchor_grid.sizes() << std::endl;
        return std::tuple<Tensor, Tensor>(grid, anchor_grid);
    }

    int nc; // number of classes
    int no; // number of outputs per anchor
    int nl; // number of detection layers
    int na; // number of anchors
    // std::vector<Tensor> grid;        // init grid
    // std::vector<Tensor> anchor_grid; // init anchor grid
    Tensor anchors; // shape(nl,na,2)
    ModuleList m;   // output conv
    bool inplace;   // use inplace ops (e.g. slice assignment)
    Tensor stride;  // default [8., 16., 32.]
    bool training;
    Tensor t_test;

    int i;              // attach index
    std::vector<int> f; // 'from' index
    std::string t;      // type, default class name
    int np;             // number params
};


ModuleList parse_model(std::string f = "C:/Users/77274/projects/MJ/libtorch-yolov5/data/yolov5s.yaml", int img_channels = 3)
{
    ModuleList module_list;
    YAML::Node config = YAML::LoadFile(f);
    std::vector<std::vector<int>> anchors;
    int nc = config["nc"].as<int>();
    float gd = config["depth_multiple"].as<float>();
    float gw = config["width_multiple"].as<float>();

    anchors = config["anchors"].as<std::vector<std::vector<int>>>();

    int na = anchors[0].size() / 2; // number of anchors
    int no = na * (nc + 5);         // number of outputs = anchors * (classes + 5)

    // std::cout << "na " << na << "no " << no << std::endl;
    std::vector<int> ch{img_channels};

    auto backbone = config["backbone"];
    for (std::size_t i = 0; i < backbone.size(); i++)
    {
        int c2;
        int from = backbone[i][0].as<int>();
        int number = backbone[i][1].as<int>();
        if (number > 1)
        {
            number = round(number * gd);
            number = std::max(number, 1);
        }

        std::string mdtype = backbone[i][2].as<std::string>();
        std::vector<int> args = backbone[i][3].as<std::vector<int>>();
        // std::cout << from << " " << number << " " << mdtype << " ";
        if (mdtype == "Conv")
        {
            int c1 = ch.back();
            c2 = args[0];
            if (c2 != no)
            {
                c2 = make_divisible(c2 * gw, 8);
            }
            std::vector<int> args_{c1, c2};
            args_.insert(args_.end(), args.begin() + 1, args.end());
            // for (size_t i = 0; i < args_.size(); i++)
            // {
            //     /* code */
            //     std::cout << args_[i] << " ";
            // }
            // std::cout << std::endl;
            {
                int c1;
                int c2;
                int k = 1;
                int s = 1;
                int p = -1;
                int g = 1;
                int d = 1;
                bool act = true;

                int n = args_.size();
                // std::cout << "args size = " << n << std::endl;
                switch (n)
                {
                case 5:
                {
                    p = args_[4];
                }
                case 4:
                {
                    s = args_[3];
                }
                case 3:
                {
                    k = args_[2];
                }
                case 2:
                {
                    c2 = args_[1];
                }
                case 1:
                {
                    c1 = args_[0];
                }
                }
                if (number > 1)
                {
                    Sequential m;
                    for (size_t j = 0; j < number; j++)
                    {
                        m->push_back(ModuleHolder<Conv>(c1, c2, k, s, p, g, d, act));
                    }
                    module_list->push_back(m);
                }
                else
                {
                    auto layer = ModuleHolder<Conv>(c1, c2, k, s, p, g, d, act);
                    layer->i = i;
                    layer->f = from;
                    layer->t = layer->name();
                    layer->np = 0;
                    auto np = layer->parameters();
                    for (const auto &p : np)
                    {
                        layer->np += p.numel();
                    }
                    module_list->push_back(layer);
                }
            }
        }
        else if (mdtype == "C3")
        {
            int c1 = ch.back();
            c2 = args[0];
            if (c2 != no)
            {
                c2 = make_divisible(c2 * gw, 8);
            }
            std::vector<int> args_{c1, c2};
            args_.insert(args_.end(), args.begin() + 1, args.end());
            args_.insert(args_.begin() + 2, number);
            // for (size_t i = 0; i < args_.size(); i++)
            // {
            //     /* code */
            //     std::cout << args_[i] << " ";
            // }
            // std::cout << std::endl;
            {
                int c1;
                int c2;
                int n = 1;
                bool shortcut = true;
                int g = 1;
                float e = 0.5;
                switch (args_.size())
                {
                case 3:
                {
                    n = args_[2];
                }
                case 2:
                {
                    c2 = args_[1];
                }
                case 1:
                {
                    c1 = args_[0];
                }
                }
                auto layer = ModuleHolder<C3>(c1, c2, n, shortcut, g, e);
                layer->i = i;
                layer->f = from;
                layer->t = layer->name();
                layer->np = 0;
                auto np = layer->parameters();
                for (const auto &p : np)
                {
                    layer->np += p.numel();
                }
                module_list->push_back(layer);
            }
        }
        else if (mdtype == "SPPF")
        {
            int c1 = ch.back();
            c2 = args[0];
            if (c2 != no)
            {
                c2 = make_divisible(c2 * gw, 8);
            }
            std::vector<int> args_{c1, c2};
            args_.insert(args_.end(), args.begin() + 1, args.end());
            // for (size_t i = 0; i < args_.size(); i++)
            // {
            //     /* code */
            //     std::cout << args_[i] << " ";
            // }
            // std::cout << std::endl;
            {
                int c1;
                int c2;
                int k = 5;
                switch (args_.size())
                {
                case 3:
                {
                    k = args_[2];
                }
                case 2:
                {
                    c2 = args_[1];
                }
                case 1:
                {
                    c1 = args_[0];
                }
                }
                auto layer = ModuleHolder<SPPF>(c1, c2, k);
                layer->i = i;
                layer->f = from;
                layer->t = layer->name();
                layer->np = 0;
                auto np = layer->parameters();
                for (const auto &p : np)
                {
                    layer->np += p.numel();
                }
                module_list->push_back(layer);
            }
        }

        if (i == 0)
            ch.clear();
        ch.push_back(c2);
        // module_list->push_back(m);
        // module_list->register_module("layer_" + std::to_string(i), m);
    }

    auto head = config["head"];
    for (std::size_t i = 0; i < head.size(); i++)
    {
        int number = head[i][1].as<int>();
        if (number > 1)
        {
            number = round(number * gd);
            number = std::max(number, 1);
        }

        std::string mdtype = head[i][2].as<std::string>();
        // std::cout << -1 << " " << number << " " << mdtype << " ";
        if (mdtype == "Conv")
        {
            int c1 = ch.back();
            std::vector<int> args = head[i][3].as<std::vector<int>>();
            // YAML::Node args = head[i][3];
            int c2 = args[0];
            if (c2 != no)
            {
                c2 = make_divisible(c2 * gw, 8);
            }
            ch.push_back(c2);
            std::vector<int> args_{c1, c2};
            args_.insert(args_.end(), args.begin() + 1, args.end());
            // for (size_t i = 0; i < args_.size(); i++)
            // {
            //     /* code */
            //     std::cout << args_[i] << " ";
            // }
            // std::cout << std::endl;
            {
                int c1;
                int c2;
                int k = 1;
                int s = 1;
                int p = -1;
                int g = 1;
                int d = 1;
                bool act = true;

                int n = args_.size();
                // std::cout << "args size = " << n << std::endl;
                switch (n)
                {
                case 5:
                {
                    p = args_[4];
                }
                case 4:
                {
                    s = args_[3];
                }
                case 3:
                {
                    k = args_[2];
                }
                case 2:
                {
                    c2 = args_[1];
                }
                case 1:
                {
                    c1 = args_[0];
                }
                }
                if (number > 1)
                {
                    Sequential m;
                    for (size_t j = 0; j < number; j++)
                    {
                        m->push_back(ModuleHolder<Conv>(c1, c2, k, s, p, g, d, act));
                    }
                    module_list->push_back(m);
                }
                else
                {
                    int from = head[i][0].as<int>();
                    auto layer = ModuleHolder<Conv>(c1, c2, k, s, p, g, d, act);
                    layer->i = i + backbone.size();
                    layer->f = from;
                    layer->t = layer->name();
                    layer->np = 0;
                    auto np = layer->parameters();
                    for (const auto &p : np)
                    {
                        layer->np += p.numel();
                    }
                    module_list->push_back(layer);
                }
            }
        }
        else if (mdtype == "nn.Upsample")
        {
            int c2 = ch.back();
            ch.push_back(c2);
            YAML::Node args = head[i][3];
            int scale_factor = args[1].as<int>();
            // std::cout << "[None"
            //           << ", " << scale_factor << ", "
            //           << "nearest]" << std::endl;
            if (number > 1)
            {
                Sequential m;
                for (size_t j = 0; j < number; j++)
                {

                    m->push_back(Upsample(UpsampleOptions()
                                              .scale_factor(std::vector<double>({double(scale_factor)}))
                                              .mode(torch::kNearest)));
                }
                module_list->push_back(m);
            }
            else
            {
                int from = head[i][0].as<int>();
                auto layer = Upsample(UpsampleOptions()
                                          .scale_factor(std::vector<double>({double(scale_factor), double(scale_factor)}))
                                          .mode(torch::kNearest));
                // layer->i = i + backbone.size();
                // layer->f = from;
                // layer->t = layer->name();
                // layer->np = 0;
                // auto np = layer->parameters();
                // for (const auto &p : np)
                // {
                //     layer->np += p.numel();
                // }
                module_list->push_back(layer);
            }
        }
        else if (mdtype == "Concat")
        {
            int c2 = 0;
            std::vector<int> from = head[i][0].as<std::vector<int>>();
            // for (size_t i = 0; i < from.size(); i++)
            // {
            //     std::cout << from[i] << " ";
            // }
            for (size_t i = 0; i < from.size(); i++)
            {
                /* code */
                if (from[i] == -1)
                    c2 += ch.back();
                // from[i] = ch.back();
                else
                    c2 += ch[from[i]];
                // from[i] = ch[from[i]];

                // std::cout << from[i] << " ";
            }
            // std::cout << "[1]" << std::endl;
            // int c2 = std::accumulate(from.begin(), from.end(), 0);
            ch.push_back(c2);
            YAML::Node args = head[i][3];
            int d = args[0].as<int>();
            if (number > 1)
            {
                Sequential m;
                for (size_t j = 0; j < number; j++)
                {
                    m->push_back(ModuleHolder<Concat>(d));
                }
                module_list->push_back(m);
            }
            else
            {
                auto layer = ModuleHolder<Concat>(d);
                layer->i = i + backbone.size();
                layer->f = from;
                layer->t = layer->name();
                layer->np = 0;
                auto np = layer->parameters();
                for (const auto &p : np)
                {
                    layer->np += p.numel();
                }
                module_list->push_back(layer);
            }
        }
        else if (mdtype == "C3")
        {
            int c1 = ch.back();
            YAML::Node args = head[i][3];
            int c2 = args[0].as<int>();
            if (c2 != no)
            {
                c2 = make_divisible(c2 * gw, 8);
            }
            ch.push_back(c2);
            std::vector<int> args_{c1, c2};
            args_.insert(args_.begin() + 2, number);
            // for (size_t i = 0; i < args_.size(); i++)
            // {
            //     /* code */
            //     std::cout << args_[i] << " ";
            // }
            // std::cout << std::endl;
            {
                int c1;
                int c2;
                int n = 1;
                bool shortcut = false;
                int g = 1;
                float e = 0.5;
                switch (args_.size())
                {
                case 3:
                {
                    n = args_[2];
                }
                case 2:
                {
                    c2 = args_[1];
                }
                case 1:
                {
                    c1 = args_[0];
                }
                }
                int from = head[i][0].as<int>();
                auto layer = ModuleHolder<C3>(c1, c2, n, shortcut, g, e);
                layer->i = i + backbone.size();
                layer->f = from;
                layer->t = layer->name();
                layer->np = 0;
                auto np = layer->parameters();
                for (const auto &p : np)
                {
                    layer->np += p.numel();
                }
                module_list->push_back(layer);
            }
        }
        else if (mdtype == "Detect")
        {
            std::vector<int> ch_;
            std::vector<int> from = head[i][0].as<std::vector<int>>();
            for (size_t i = 0; i < from.size(); i++)
            {
                ch_.push_back(ch[from[i]]);
            }
            auto layer = ModuleHolder<Detect>(nc, anchors, ch_, true);
            layer->i = i + backbone.size();
            layer->f = from;
            layer->t = layer->name();
            layer->np = 0;
            auto np = layer->parameters();
            for (const auto &p : np)
            {
                layer->np += p.numel();
            }
            module_list->push_back(layer);
            // std::cout << layer->t <<" " << layer->np << std::endl;
        }
    }

    // std::cout << module_list << std::endl;
    // auto inputs = torch::randn({1,3,640,640});
    // auto outputs =  module_list->forward(inputs);
    return module_list;
}

struct DetectionModel : Module
{
    DetectionModel(std::string cfg = "", int ch = 3)
    {
        module_list = parse_model(cfg, ch);
        register_module("model", module_list);
        int s = 256;
        auto inputs = torch::zeros({1, ch, s, s});
        auto o = forward(inputs);
        std::vector<int> stride;
        for (size_t i = 0; i < o.size(); i++)
        {
            int v = s / o[i].size(3);
            stride.push_back(v);
        }
        auto module = module_list->ptr<Module>(module_list->size() - 1);
        auto m = module->as<Detect>();
        m->stride = torch::tensor(stride, torch::kFloat32);
        // auto  anchors = m->named_buffers()["anchors"];
        m->anchors = m->anchors / m->stride.view({-1, 1, 1});
        // std::cout <<"m->stride " << m->stride << std::endl;
        // std::cout << "m->anchors " << m->named_buffers()["anchors"] << std::endl;
    }

    std::vector<Tensor> forward(torch::Tensor x)
    {
        return _forward_once(x);
    }

    std::vector<Tensor> _forward_once(torch::Tensor x)
    {
        std::vector<Tensor> outputs{x};
        Tensor O;
        for (size_t i = 0; i < module_list->size(); i++)
        {
            auto module = module_list->ptr<Module>(i);
            auto nm = module->name();
            if (nm == "Conv")
            {
                auto m = module->as<Conv>();
                if (m)
                {
                    auto inputs = outputs.back();
                    O = m->forward(inputs);
                }
            }
            else if (nm == "C3")
            {
                auto m = module->as<C3>();
                if (m)
                {
                    auto inputs = outputs.back();
                    O = m->forward(inputs);
                }
            }
            else if (nm == "SPPF")
            {
                auto m = module->as<SPPF>();
                if (m)
                {
                    auto inputs = outputs.back();
                    O = m->forward(inputs);
                }
            }
            else if (nm == "Concat")
            {
                std::vector<Tensor> inputs;
                auto m = module->as<Concat>();
                for (size_t i = 0; i < m->f.size(); i++)
                {
                    if (m->f[i] == -1)
                    {
                        auto inp = outputs.back();
                        inputs.push_back(inp);
                    }
                    else
                    {
                        auto inp = outputs[m->f[i]];
                        inputs.push_back(inp);
                    }
                }
                O = m->forward(inputs);
            }
            else if (nm == "torch::nn::UpsampleImpl")
            {

                auto m = module->as<Upsample>();
                if (m)
                {
                    auto inputs = outputs.back();
                    O = m->forward(inputs);
                }
            }
            else if (nm == "Detect")
            {
                std::vector<Tensor> inputs;
                auto m = module->as<Detect>();
                for (size_t i = 0; i < m->f.size(); i++)
                {
                    if (m->f[i] == -1)
                    {
                        auto inp = outputs.back();
                        inputs.push_back(inp);
                    }
                    else
                    {
                        auto inp = outputs[m->f[i]];
                        // std::cout << "inp=" << inp.sizes() << std::endl;
                        inputs.push_back(inp);
                    }
                }
                auto outp = m->forward(inputs);
                // std::cout << "outp=" << outp[0].sizes() << std::endl;
                return outp;
            }

            if (i == 0)
                outputs.clear();
            // std::cout << "i= " << i << " " << nm << " " << O.sizes() << std::endl;
            outputs.emplace_back(O);
        }
    }

    void train(bool on = true)
    {
        auto module = module_list->ptr<Module>(module_list->size() - 1);
        auto m = module->as<Detect>();
        m->training = on;

        Module::train(on);
    }

    void load_weights(std::string weight_file)
    {
        int idx = 0;
        std::ifstream fs(weight_file, std::ios_base::binary);
        if (!fs)
        {
            throw std::runtime_error("No weight file for Darknet!");
        }
        std::vector<torch::Tensor> g0, g1, g2;
        for (auto &layer : named_modules())
        {
            auto m_name = layer.key();
            auto m_v = layer.value();

            if (auto conv = m_v->as<torch::nn::Conv2dImpl>())
            {
                auto nb = conv->weight.element_size() * conv->weight.numel();
                std::cout << idx++ << " " << conv->name() << " " << conv->weight.sizes() << " " << nb << std::endl;
                fs.read(static_cast<char *>(conv->weight.data_ptr()), nb);
                if (conv->bias.size(0))
                {
                    auto nb = conv->bias.element_size() * conv->bias.numel();
                    std::cout << idx++ << " " << conv->name() << " " << conv->bias.sizes() << " " << nb << std::endl;
                    fs.read(static_cast<char *>(conv->bias.data_ptr()), nb);
                }
            }
            else if (auto bn = m_v->as<torch::nn::BatchNorm2dImpl>())
            {
                auto nb = bn->weight.element_size() * bn->weight.numel();
                std::cout << idx++ << " " << bn->name() << " " << bn->weight.sizes() << " " << nb << std::endl;
                fs.read(static_cast<char *>(bn->weight.data_ptr()), nb);

                nb = bn->bias.element_size() * bn->bias.numel();
                std::cout << idx++ << " " << bn->name() << " " << bn->bias.sizes() << " " << nb << std::endl;
                fs.read(static_cast<char *>(bn->bias.data_ptr()), nb);

                nb = bn->running_mean.element_size() * bn->running_mean.numel();
                std::cout << idx++ << " " << bn->name() << " " << bn->running_mean.sizes() << " " << nb << std::endl;
                fs.read(static_cast<char *>(bn->running_mean.data_ptr()), nb);

                nb = bn->running_var.element_size() * bn->running_var.numel();
                std::cout << idx++ << " " << bn->name() << " " << bn->running_var.sizes() << " " << nb << std::endl;
                fs.read(static_cast<char *>(bn->running_var.data_ptr()), nb);
            }
        }
    }

    ModuleList module_list;
    std::map<std::string, float> hyp;
};
#endif