#ifndef METRICS_H
#define METRICS_H
#include <torch/torch.h>
#include <cmath>

using namespace torch;
using namespace torch::nn;
using namespace torch::indexing;

torch::Tensor interp(torch::Tensor x, torch::Tensor xp, torch::Tensor fp)
{
    if (xp.size(0) < 2)
        return xp.expand(x.size(0));

    auto m = (fp.index({Slice(1)}) - fp.index({Slice(c10::nullopt, -1)})) / (xp.index({Slice(1)}) - xp.index({Slice(c10::nullopt, -1)}));
    m = m.nan_to_num(0, 0, 0);
    auto b = fp.index({Slice(c10::nullopt, -1)}) - m * xp.index({Slice(c10::nullopt, -1)});

    auto indicies = torch::ge(x.index({Slice(), c10::nullopt}), xp.index({c10::nullopt, Slice()}));
    indicies = torch::sum(indicies, 1) - 1;
    indicies = torch::clamp(indicies, 0, m.size(0) - 1);

    auto ret = m.index({indicies}) * x + b.index({indicies});
    return ret;
}

torch::Tensor unique(torch::Tensor x)
{
    auto [unique, inverse, counts] = at::_unique2(x, false, true, true);
    auto decimals = torch::arange(inverse.numel(), inverse.device()) / inverse.numel();
    auto inv_sorted = (inverse + decimals).argsort();
    auto tot_counts = torch::cat({counts.new_zeros({1}), counts.cumsum(0)}).index({Slice(c10::nullopt, -1)});
    auto index = inv_sorted.index({tot_counts});
    return index;
}

torch::Tensor smooth(torch::Tensor y, float f = 0.05)
{
    int nf = std::round(y.size(0) * f * 2) / 2 + 1;
    auto p = torch::ones({int(nf / 2)}, y.device());
    auto yp = torch::cat({p * y[0], y, p * y[-1]}, 0);
    auto w = torch::ones({nf}, y.device()) / nf;
    return torch::nn::functional::conv1d(yp.view({1, 1, -1}), w.view({1, 1, -1})).view({-1});
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> compute_ap(torch::Tensor recall, torch::Tensor precision)
{
    // std::cout << "-------------------------compute_ap" << std::endl;
    // std::cout << recall.sizes() << recall.options() << std::endl;
    // std::cout << precision.sizes() << precision.options() << std::endl;

    auto mrec = torch::cat({torch::tensor({0.0}, recall.device()), recall, torch::tensor({1.0}, recall.device())});
    auto mpre = torch::cat({torch::tensor({1.0}, recall.device()), precision, torch::tensor({0.0}, recall.device())});
    // std::cout << mrec.sizes() << std::endl;
    // std::cout << mpre.sizes() << std::endl;
    auto [accumulated_max, accumulated_idx] = torch::cummax(torch::flip(mpre, 0), 0);
    mpre = torch::flip(accumulated_max, 0);
    // std::cout << "-------------------------interp 0" << std::endl;
    // method = 'interp'
    std::string method = "interp";
    torch::Tensor ap;
    if (method == "interp")
    {
        // std::cout << "-------------------------interp 1" << std::endl;
        auto x = torch::linspace(0, 1, 101, recall.device());
        auto tmp0 = interp(x, mrec, mpre);
        ap = torch::trapz(tmp0, x);
        bool isnan = ap.isnan().any().item<bool>();
        if (isnan)
        {
            std::cout << "recall = " << recall << std::endl;
            std::cout << "tmp0 = " << tmp0 << std::endl;
            std::cout << "mrec = " << mrec << std::endl;
            std::cout << "mpre = " << mpre << std::endl;
            abort();
        }

        // ap = torch::trapz(interp(x, mrec, mpre), x);
        // std::cout << ap << std::endl;
        // std::cout << "-------------------------interp 2" << std::endl;
    }
    // std::cout << "-------------------------" << std::endl;
    return std::make_tuple(ap, mpre, mrec);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> ap_per_class(torch::Tensor tp, torch::Tensor conf, torch::Tensor pred_cls, torch::Tensor target_cls, float eps = 1e-16)
{
    // std::cout << "-------------------------ap_per_class" << std::endl;
    auto i = at::argsort(-conf);
    tp = tp.index({i});
    conf = conf.index({i});
    pred_cls = pred_cls.index({i});
    // std::cout << "-------------------------0" << std::endl;
    auto [unique_classes, inverse, nt] = at::_unique2(target_cls, false, true, true);
    int nc = unique_classes.size(0);

    auto px = torch::linspace(0, 1, 1000).to(tp.device());
    // torch::Tensor py;
    auto ap = torch::zeros({nc, tp.size(1)}).to(tp.device());
    auto p = torch::zeros({nc, 1000}).to(tp.device());
    auto r = torch::zeros({nc, 1000}).to(tp.device());
    // std::cout << "-------------------------1" << std::endl;
    for (int ci = 0; ci < unique_classes.numel(); ci++)
    {
        auto c = unique_classes[ci];
        auto i = pred_cls == c;
        auto n_l = nt[ci].item<int>();
        auto n_p = i.sum().item<int>();
        // std::cout << n_l << " " << n_p << std::endl;
        if (n_p == 0 || n_l == 0)
            continue;
        // std::cout << "-------------------------2" << std::endl;
        auto fpc = (1 - tp.index({i}).to(torch::kF32)).cumsum(0);
        // std::cout << "fpc = " << fpc.isnan().any().item<bool>() << std::endl;
        // std::cout << "-------------------------3" << std::endl;
        auto tpc = tp.index({i}).cumsum(0).to(torch::kF32);
        // std::cout << "tpc = " <<  tpc << std::endl;
        // std::cout << "tpc = " << tpc.isnan().any().item<bool>() << std::endl;
        // std::cout << "-------------------------4" << std::endl;
        auto recall = tpc / (n_l + eps);
        // std::cout << recall << std::endl;
        // std::cout << "recall = " << recall.isnan().any().item<bool>() << std::endl;
        // std::cout << "-------------------------4" << std::endl;
        r[ci] = interp(-px, -conf.index({i}), recall.index({Slice(), 0})); //, torch::tensor({0.}, recall.device())
        // std::cout << "r[ci] = " << r[ci].isnan().any().item<bool>() << std::endl;
        // std::cout << "-------------------------5" << std::endl;
        auto precision = tpc / (tpc + fpc);
        // std::cout << "precision = " << precision.isnan().any().item<bool>() << std::endl;
        p[ci] = interp(-px, -conf.index({i}), precision.index({Slice(), 0})); //, torch::tensor({1.}, recall.device())
        // std::cout << "p[ci] = " << p[ci].isnan().any().item<bool>() << std::endl;
        // std::cout << "-------------------------6" << std::endl;
        for (int j = 0; j < tp.size(1); j++)
        {
            // std::cout << "-------------------------7" << std::endl;
            auto [v, mpre, mrec] = compute_ap(recall.index({Slice(), j}), precision.index({Slice(), j}));
            ap.index({ci, j}) = v;
            // std::cout << "ap.index({ci, j}) = " << ap.index({ci, j}).isnan().any().item<bool>() << std::endl;
        }
    }
    // std::cout << "-------------------------8" << std::endl;
    auto f1 = 2 * p * r / (p + r + eps);
    // std::cout << "-------------------------9" << std::endl;
    i = smooth(f1.mean(0), 0.1).argmax();
    // std::cout << "-------------------------10" << std::endl;
    p = p.index({Slice(), i});
    // std::cout << "-------------------------11" << std::endl;
    r = r.index({Slice(), i});
    // std::cout << "-------------------------12" << std::endl;
    f1 = f1.index({Slice(), i});
    // std::cout << "-------------------------13" << std::endl;
    // std::cout << "r = " << r << std::endl;
    // std::cout << "nt = " << nt << std::endl;
    tp = (r * nt).round();
    // std::cout << "-------------------------14" << std::endl;
    auto fp = (tp / (p + eps) - tp).round();
    // std::cout << "-------------------------15" << std::endl;
    return std::make_tuple(tp, fp, p, r, f1, ap, unique_classes.to(torch::kInt32));
}

torch::Tensor box_iou(torch::Tensor box1, torch::Tensor box2, float eps = 1e-7)
{
    auto a = box1.unsqueeze(1).chunk(2, 2);
    auto a1 = a[0], a2 = a[1];
    auto b = box2.unsqueeze(0).chunk(2, 2);
    auto b1 = b[0], b2 = b[1];
    auto inter = (torch::min(a2, b2) - torch::max(a1, b1)).clamp_(0).prod(2);
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps);
}

// void _linalg_cond_check_ord(c10::variant<Scalar, c10::string_view> ord_variant) {
//   if (ord_variant.index() == 0) {
//     Scalar* ord = c10::get_if<Scalar>(&ord_variant);
//     double abs_ord = std::abs(ord->toDouble());
//     TORCH_CHECK(abs_ord == 2.0 || abs_ord == 1.0 || abs_ord == INFINITY,
//       "linalg.cond got an invalid norm type: ", ord->toDouble());
//   } else if (ord_variant.index() == 1) {
//     c10::string_view* ord = c10::get_if<c10::string_view>(&ord_variant);
//     TORCH_CHECK(*ord == "fro" || *ord == "nuc",
//       "linalg.cond got an invalid norm type: ", *ord);
//   } else {
//     TORCH_CHECK(false,
//       "linalg.cond: something went wrong while checking the norm type");
//   }
// }

class ConfusionMatrix
{
public:
    ConfusionMatrix(int nc, float conf = 0.25, float iou_thres = 0.45) : m_nc(nc), m_conf(conf), m_iou_thres(iou_thres)
    {
        m_matrix = torch::zeros({nc + 1, nc + 1});
    }
    //[N, 6] [M, 5]
    void process_batch(c10::optional<torch::Tensor> detections_, torch::Tensor labels)
    {
        if (detections_ == c10::nullopt)
        {
            return;
        }
        torch::Tensor detections = detections_.value();
        detections = detections.index({detections.index({Slice(), 4}) > m_conf});
        auto gt_classes = labels.index({Slice(), 0}).to(torch::kInt32);
        auto detection_classes = detections.index({Slice(), 5}).to(torch::kInt32);
        auto iou = box_iou(labels.index({Slice(), Slice(1)}), detections.index({Slice(), Slice(None, 4)}));
        torch::Tensor matches;
        auto x = torch::where(iou > m_iou_thres);
        if (x[0].size(0))
        {
            matches = torch::cat({torch::stack(x, 1), iou.index({x[0], x[1]}).index({Slice(), None})}, 1);
            if (x[0].size(0) > 1)
            {
                matches = matches.index({matches.index({Slice(), 2}).argsort(-1, true)});
                matches = matches.index({unique(matches.index({Slice(), 1}))});
                matches = matches.index({matches.index({Slice(), 2}).argsort(-1, true)});
                matches = matches.index({unique(matches.index({Slice(), 0}))});
            }
        }
        else
        {
            matches = torch::zeros({0, 3});
        }

        bool n = matches.size(0) > 0;
        auto vec_m = matches.permute({1, 0}).chunk(3, 0);
        auto m0 = vec_m[0], m1 = vec_m[1];
        for (int i = 0; i < gt_classes.numel(); i++)
        {
            /* code */
            auto gc = gt_classes[i];
            torch::Tensor j = (m0 == i);
            if (n && j.sum().item<int>() == 1)
            {
                auto t0 = m1.index({j});
                auto t1 = detection_classes.index({t0});
                m_matrix.index({t1, gc}) += 1;
            }
            else
            {
                m_matrix.index({m_nc, gc}) += 1;
            }
        }

        if (n)
        {
            for (int i = 0; i < detection_classes.numel(); i++)
            {
                /* code */
                auto dc = detection_classes[i];
                auto j = m1 == i;
                if (!(j.any().item<bool>()))
                    m_matrix.index({dc, m_nc}) += 1;
            }
        }
    }

    torch::Tensor matrix()
    {
        return m_matrix;
    }

    std::tuple<torch::Tensor, torch::Tensor> tp_fp()
    {
        auto tp = m_matrix.diagonal();
        auto fp = m_matrix.sum(1) - tp;
        return std::make_tuple(tp.index({Slice(c10::nullopt, -1)}), fp.index({Slice(c10::nullopt, -1)}));
    }

private:
    int m_nc;
    float m_conf;
    float m_iou_thres;
    torch::Tensor m_matrix;
};

Tensor bbox_iou(Tensor box1, Tensor box2, bool xywh = true, bool GIoU = false, bool DIoU = false, bool CIoU = false, float eps = 1e-7)
{
    Tensor x1, y1, w1, h1, x2, y2, w2, h2;
    Tensor b1_x1, b1_x2, b1_y1, b1_y2, b2_x1, b2_x2, b2_y1, b2_y2;
    if (xywh)
    {
        auto vec0 = box1.chunk(4, -1);
        x1 = vec0[0];
        y1 = vec0[1];
        w1 = vec0[2];
        h1 = vec0[3];
        auto vec1 = box2.chunk(4, -1);
        x2 = vec1[0];
        y2 = vec1[1];
        w2 = vec1[2];
        h2 = vec1[3];

        auto w1_ = w1 / 2;
        auto h1_ = h1 / 2;
        auto w2_ = w2 / 2;
        auto h2_ = h2 / 2;

        b1_x1 = x1 - w1_;
        b1_x2 = x1 + w1_;
        b1_y1 = y1 - h1_;
        b1_y2 = y1 + h1_;

        b2_x1 = x2 - w2_;
        b2_x2 = x2 + w2_;
        b2_y1 = y2 - h2_;
        b2_y2 = y2 + h2_;
        // std::cout << "x1 =" << x1.sizes() << std::endl;
        // std::cout << "y1 =" << y1.sizes() << std::endl;
        // std::cout << "w1 =" << w1.sizes() << std::endl;
        // std::cout << "h1 =" << h1.sizes() << std::endl;
    }
    else
    {
        auto vec0 = box1.chunk(4, -1);
        b1_x1 = vec0[0];
        b1_y1 = vec0[1];
        b1_x2 = vec0[2];
        b1_y2 = vec0[3];
        auto vec1 = box2.chunk(4, -1);
        b2_x1 = vec1[0];
        b2_y1 = vec1[1];
        b2_x2 = vec1[2];
        b2_y2 = vec1[3];
        w1 = b1_x2 - b1_x1;
        h1 = b1_y2 - b1_y1 + eps;
        w2 = b2_x2 - b2_x1;
        h2 = b2_y2 - b2_y1 + eps;
    }

    auto t0_w = torch::min(b1_x2, b2_x2) - torch::max(b1_x1, b2_x1);
    t0_w = t0_w.clamp(0);
    auto t0_h = torch::min(b1_y2, b2_y2) - torch::max(b1_y1, b2_y1);
    t0_h = t0_h.clamp(0);

    auto inter = t0_w * t0_h;
    auto unions = w1 * h1 + w2 * h2 - inter + eps;

    auto iou = inter / unions;
    if (CIoU || DIoU || GIoU)
    {
        auto cw = torch::max(b1_x2, b2_x2) - torch::min(b1_x1, b2_x1);
        auto ch = torch::max(b1_y2, b2_y2) - torch::min(b1_y1, b2_y1);
        if (CIoU || DIoU)
        {
            auto c2 = cw.pow(2) + ch.pow(2) + eps;
            auto rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)) / 4;
            if (CIoU)
            {
                auto v = (4 / (M_PI * M_PI)) * torch::pow(torch::atan(w2 / h2) - torch::atan(w1 / h1), 2);
                auto alpha = v / (v - iou + (1 + eps));
                return iou - (rho2 / c2 + v * alpha);
            }
            return iou - rho2 / c2;
        }
        auto c_area = cw * ch + eps;
        return iou - (c_area - unions) / c_area;
    }
    return iou;
}

#endif