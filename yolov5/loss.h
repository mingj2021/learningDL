#ifndef LOSS_H
#define LOSS_H

#include <torch/torch.h>
#include "yolo.h"
#include "metrics.h"
#include <torch/script.h>
using namespace torch;
using namespace torch::nn;
using namespace torch::indexing;

std::tuple<float, float> smooth_BCE(float eps = 0.1)
{
    return std::tuple<float, float>(1.0 - 0.5 * eps, 0.5 * eps);
}

struct FocalLoss : Module
{
    FocalLoss(BCEWithLogitsLoss loss_fcn_, float gamma_ = 1.5, float alpha_ = 0.25)
    {
        loss_fcn = loss_fcn_;
        gamma = gamma_;
        alpha = alpha_;
        auto r = loss_fcn->options.reduction();
        reduction = torch::enumtype::reduction_get_enum<torch::nn::BCEWithLogitsLossOptions::reduction_t>(r);
        loss_fcn->options.reduction(torch::kNone);
    }

    torch::Tensor forward(torch::Tensor pred_, torch::Tensor true_)
    {
        auto loss = loss_fcn(pred_, true_);
        auto pred_prob = pred_.sigmoid();
        auto p_t = true_ * pred_prob + (1 - true_) * (1 - pred_prob);
        auto alpha_factor = true_ * alpha + (1 - true_) * (1 - alpha);
        auto modulating_factor = 1.0 - p_t;
        modulating_factor = modulating_factor.pow(gamma);
        loss *= alpha_factor * modulating_factor;

        torch::Tensor x;
        if (reduction == at::Reduction::Reduction::Mean)
        {
            x = loss.mean();
        }
        else if (reduction == at::Reduction::Reduction::Sum)
        {
            x = loss.sum();
        }
        else
        {
            x = loss;
        }

        return x;
    }

    BCEWithLogitsLoss loss_fcn; // must be nn.BCEWithLogitsLoss()
    float gamma;
    float alpha;
    at::Reduction::Reduction reduction;
};

struct ComputeLoss : Module
{
    torch::Device device;
    std::map<std::string, float> hyp;
    BCEWithLogitsLoss BCEcls, BCEobj;
    float cp, cn;
    std::vector<float> balance;
    int ssi;
    float gr;
    int na;
    int nc;
    int nl;
    torch::Tensor anchors;
    bool sort_obj_iou;
    bool autobalance;
    ComputeLoss(ModuleHolder<DetectionModel> model, bool autobalance_ = false) : device(torch::kCPU), sort_obj_iou(false), autobalance(autobalance_)
    {
        device = model->parameters()[0].device();
        hyp = model->hyp;
        auto weight = torch::tensor({hyp["cls_pw"]});
        BCEcls = BCEWithLogitsLoss(BCEWithLogitsLossOptions().pos_weight(weight));
        BCEcls->to(device);
        weight = torch::tensor({hyp["obj_pw"]});
        BCEobj = BCEWithLogitsLoss(BCEWithLogitsLossOptions().pos_weight(weight));
        BCEobj->to(device);
        float eps = 0.0;

        if (hyp.find("label_smoothing") != hyp.end())
            eps = hyp["label_smoothing"];
        auto s = smooth_BCE(eps);
        // std::cout << "eps = " << eps << std::endl;
        cp = std::get<0>(s);
        // std::cout << "cp = " << cp << std::endl;
        cn = std::get<1>(s);
        // std::cout << "cn = " << cn << std::endl;

        // Focal loss
        float g = hyp["fl_gamma"];
        if (g > 0.0)
        {
            // BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
        }
        auto module = model->module_list->ptr<Module>(model->module_list->size() - 1);
        auto m = module->as<Detect>();

        if (m->nl == 3)
        {
            balance = std::vector<float>({4.0, 1.0, 0.4});
        }
        else
        {
            balance = std::vector<float>({4.0, 1.0, 0.25, 0.06, 0.02});
        }

        if (autobalance)
        {
            ssi = 1;
        }
        else
        {
            ssi = 0;
        }
        gr = 1.0;
        na = m->na;
        nc = m->nc;
        nl = m->nl;
        anchors = m->anchors;
    }

    std::vector<torch::Tensor> operator()(std::vector<torch::Tensor> p, torch::Tensor targets)
    {
        // torch::jit::script::Module container = torch::jit::load("C:/Users/77274/projects/MJ/libtorch-yolov5/data/build_targets.pt");

        ///////////////////////////////////////////////////////////////////////////////
        auto lcls = torch::zeros({1}, device);
        auto lbox = torch::zeros({1}, device);
        auto lobj = torch::zeros({1}, device);
        auto tobj = torch::zeros({1}, device);

        auto [tcls, tbox, indices, anchors] = build_targets(p, targets);
        // std::cout << "build_targets-------- " << std::endl;
        // Losses
        for (size_t i = 0; i < p.size(); i++)
        {
            auto pi = p[i]; // layer index, layer predictions
            // image, anchor, gridy, gridx
            auto b = indices[i][0];
            auto a = indices[i][1];
            auto gj = indices[i][2];
            auto gi = indices[i][3];
            tobj = torch::zeros({pi.size(0), pi.size(1), pi.size(2), pi.size(3)}).to(device);
            int n = b.size(0);
            if (n)
            {
                auto vec0 = pi.index({b, a, gj, gi}).split({2, 2, 1, nc}, 1); // target-subset of predictions
                // std::cout << vec0.size() << std::endl;
                auto pxy = vec0[0];
                auto pwh = vec0[1];
                auto pcls = vec0[3];

                // Regression
                pxy = pxy.sigmoid() * 2 - 0.5;
                pwh = pwh.sigmoid().mul(2).pow(2) * anchors[i];
                auto pbox = torch::cat({pxy, pwh}, 1); // predicted box
                auto iou = bbox_iou(pbox, tbox[i], true, false, false, true).squeeze();
                lbox += (1.0 - iou).mean();
                // std::cout << "iou= " << iou.sizes() << std::endl;
                // std::cout << "lbox= " << lbox << std::endl;

                // Objectness
                iou = iou.detach().clamp(0);
                if (sort_obj_iou)
                {
                    auto j = iou.argsort();
                    b = b[j];
                    a = a[j];
                    gj = gj[j];
                    gi = gi[j];
                    iou = iou[j];
                }
                if (gr < 1)
                {
                    iou = (1.0 - gr) + gr * iou;
                }
                // tobj.index_put_({b, a, gj, gi}, iou);
                // std::cout << "tobj=" << tobj.index({b,a,gj,gi}) << std::endl;
                for (size_t i = 0; i < b.numel(); i++)
                {
                    /* code */
                    tobj.index({b[i],a[i],gj[i],gi[i]}) = iou[i];
                }

                // Classification
                if (nc > 1)
                {
                    auto t = torch::full_like(pcls, cn);
                    // std::cout << "t1=" << t.sizes() << std::endl;
                    t.index_put_({torch::arange(n), tcls[i]}, cp);
                    // std::cout << "t2=" << t.sizes() << std::endl;
                    lcls += BCEcls(pcls, t);
                    // std::cout << "lcls=" << lcls << std::endl;
                }
            }
            auto obji = BCEobj(pi.index({"...", 4}), tobj);
            lobj += obji * balance[i];
            // if (i == 2)
            // {
            //     torch::jit::script::Module container = torch::jit::load("C:/Users/77274/projects/MJ/libtorch-yolov5/data/build_targets.pt");
            //     // auto py_pxy0 = container.attr("pxy0").toTensor();
            //     // std::cout << "pxy0 " << at::equal(py_pxy0, pxy) << std::endl;
            //     // auto py_pwh10 = container.attr("pwh10").toTensor();
            //     // std::cout << "pxy0 " << at::equal(py_pwh10, pwh) << std::endl;
            //     auto py_tt = container.attr("lobj2").toTensor();
            //     auto py_tt2 = lobj;
            //     std::cout << "obji " << at::equal(py_tt, py_tt2) << std::endl;

            //     // auto py_anchors0 = container.attr("anchors0").toTensor();
            //     // std::cout << "py_anchors0 " << at::equal(py_anchors0, anchors[i]) << std::endl;
            //     // std::cout << anchors[i] << std::endl;

            //     // auto py_pbox0 = container.attr("pbox0").toTensor();
            //     // std::cout << "pbox0 " << at::equal(py_pbox0, pbox) << std::endl;

            //     // auto py_iou0 = container.attr("iou0").toTensor();
            //     // std::cout << "iou0 " << at::equal(py_iou0, iou) << std::endl;

            //     // auto py_lbox0 = container.attr("lbox0").toTensor();
            //     // std::cout << "lbox0 " << at::equal(py_lbox0, lbox) << std::endl;

            //     std::cout << py_tt << std::endl;
            //     std::cout << "------------------------------------------" << std::endl;
            //     std::cout << py_tt2 << std::endl;
            // }
            if (autobalance)
            {
                // std::cout << "obji.detach().item<float>() " << obji.detach().item<float>() << std::endl;
                balance[i] = balance[i] * 0.9999 + 0.0001 / obji.detach().item<float>();
            }
        }

        if (autobalance)
        {
            for (size_t i = 0; i < balance.size(); i++)
            {
                balance[i] = balance[i] / balance[ssi];
            }
        }
        lbox *= hyp["box"];
        lobj *= hyp["obj"];
        lcls *= hyp["cls"];
        auto bs = tobj.size(0);
        std::vector<torch::Tensor> outps;
        // auto ttt1 = (lbox + lobj + lcls) * bs;
        // auto ttt2 = torch::cat({lbox, lobj, lcls});
        // std::cout << "ttt2= " <<ttt2.sizes() << std::endl;
        outps.push_back((lbox + lobj + lcls) * bs);
        outps.push_back(torch::cat({lbox, lobj, lcls}).detach());
        return outps;
    }

    std::tuple<std::vector<Tensor>, std::vector<Tensor>, std::vector<std::vector<Tensor>>, std::vector<Tensor>> build_targets(std::vector<torch::Tensor> p, torch::Tensor targets)
    {
        // std::cout << "------------------" << std::endl;
        std::vector<std::vector<Tensor>> oupts_indices;
        std::vector<Tensor> oupts_tbox;
        std::vector<Tensor> oupts_anch;
        std::vector<Tensor> oupts_tcls;
        int nt = targets.size(0);             // number of targets
        auto gain = torch::ones({7}, device); // normalized to gridspace gain
        auto ai = torch::arange(na, device).to(torch::kFloat).view({na, 1}).repeat({1, nt});
        targets = torch::cat({targets.repeat({na, 1, 1}), ai.unsqueeze(ai.dim())}, 2); // append anchor indices
        float g = 0.5;
        auto off = torch::tensor({{0, 0},
                                  {1, 0},
                                  {0, 1},
                                  {-1, 0},
                                  {0, -1}},
                                 device)
                       .to(torch::kFloat) *
                   g;
        for (size_t i = 0; i < nl; i++)
        {
            auto anch = anchors[i].to(device);
            // std::cout << "anch = " << anch.device() << std::endl;
            auto shape = p[i].sizes();
            gain.index({Slice(2, 6)}) = torch::tensor(shape).to(device).index_select(0, torch::tensor({3, 2, 3, 2}).to(device)); // xyxy gain
            // std::cout << gain << std::endl;
            // Match targets to anchors
            auto t = targets * gain; // shape(3,n,7)
            if (nt)
            {
                // Matches
                auto r = t.index({"...", Slice(4, 6)}) / anch.index({Slice(), None});
                auto [j, k] = torch::max(r, 1 / r).max(2);
                j = j < hyp["anchor_t"]; // compare
                t = t.index({j});        // filter
                // std::cout << t.sizes() << std::endl;

                // Offsets
                auto gxy = t.index({Slice(), Slice(2, 4)}); // grid xy
                auto gxi = gain.index_select(0, torch::tensor({2, 3}).to(device)) - gxy;
                auto Temp = (gxy % 1 < g) & (gxy > 1);
                Temp = Temp.t();
                j = Temp.index({0, Slice()});
                k = Temp.index({1, Slice()});
                // std::cout << "j =" << j.sizes() << std::endl;
                // std::cout << "k =" <<k.sizes() << std::endl;
                Temp = (gxi % 1 < g) & (gxi > 1);
                Temp = Temp.t();
                auto l = Temp.index({0, Slice()});
                auto m = Temp.index({1, Slice()});
                j = torch::stack({torch::ones_like(j), j, k, l, m});
                t = t.repeat({5, 1, 1}).index({j});
                auto offsets = torch::zeros_like(gxy).index({None}) + off.index({Slice(), None});
                offsets = offsets.index({j});
                // std::cout << "offsets" << offsets.sizes() << std::endl;
                auto vec = t.chunk(4, 1); //(image, class), grid xy, grid wh, anchors
                auto &bc = vec[0];
                gxy = vec[1];
                auto &gwh = vec[2];
                auto &a = vec[3];
                a = a.to(torch::kInt32).view({-1}); // anchors
                // std::cout << "a= " << a.device() << std::endl;
                bc = bc.to(torch::kInt32).t();
                // std::cout << "bc= " << bc.device() << std::endl;
                auto b = bc.index({0, Slice()});
                // std::cout << "b= " << b.device() << std::endl;
                auto c = bc.index({1, Slice()});
                // std::cout << "c= " << c.device() << std::endl;
                auto gij = gxy - offsets;
                gij = gij.to(torch::kInt32);
                auto gij_ = gij.t();
                // std::cout << "gij= " << gij.device() << std::endl;
                auto gi = gij_.index({0, Slice()});
                // std::cout << "gi= " << gi.device() << std::endl;
                auto gj = gij_.index({1, Slice()});
                // std::cout << "gj= " << gj.device() << std::endl;

                // std::cout << "anchors[a]= " << anch.index({a}).device() << std::endl;
                oupts_indices.push_back(std::vector<Tensor>({b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)})); // image, anchor, grid
                oupts_tbox.push_back(torch::cat({gxy - gij, gwh}, 1));                                                        // box
                oupts_anch.push_back(anch.index({a}));                                                                        // anchors
                oupts_tcls.push_back(c);                                                                                      // class
            }
            else
            {
                t = targets[0];
                int offsets = 0;

                auto vec = t.chunk(4, 1); //(image, class), grid xy, grid wh, anchors
                auto &bc = vec[0];
                auto gxy = vec[1];
                auto &gwh = vec[2];
                auto &a = vec[3];
                a = a.to(torch::kInt32).view({-1}); // anchors
                // std::cout << "a= " << a.sizes() << std::endl;
                bc = bc.to(torch::kInt32).t();
                // std::cout << "bc= " << bc.sizes() << std::endl;
                auto b = bc.index({0, Slice()});
                // std::cout << "b= " << b.sizes() << std::endl;
                auto c = bc.index({1, Slice()});
                // std::cout << "c= " << c.sizes() << std::endl;
                auto gij = gxy - offsets;
                auto gij_ = gij.t();
                // std::cout << "gij= " << gij.sizes() << std::endl;
                auto gi = gij_.index({0, Slice()});
                // std::cout << "gi= " << gi.sizes() << std::endl;
                auto gj = gij_.index({1, Slice()});
                // std::cout << "gj= " << gj.sizes() << std::endl;

                // std::cout << "anchors[a]= " << anch.index({a}).sizes() << std::endl;
                oupts_indices.push_back(std::vector<Tensor>({b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)})); // image, anchor, grid
                oupts_tbox.push_back(torch::cat({gxy - gij, gwh}, 1));                                                        // box
                oupts_anch.push_back(anch.index({a}));                                                                        // anchors
                oupts_tcls.push_back(c);                                                                                      // class
            }
        }
        return std::tuple<std::vector<Tensor>, std::vector<Tensor>, std::vector<std::vector<Tensor>>, std::vector<Tensor>>(oupts_tcls, oupts_tbox, oupts_indices, oupts_anch);
    }
};

#endif