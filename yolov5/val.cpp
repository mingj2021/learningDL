#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <yaml-cpp/yaml.h>
#include "yolo.h"
#include "loss.h"
#include <sstream>
#include "dataloaders.h"

using namespace torch::indexing;

torch::Tensor process_batch(torch::Tensor detections, torch::Tensor labels, torch::Tensor iouv)
{
    // std::cout << "--------------------------process_batch" << std::endl;
    auto correct = torch::zeros({detections.size(0), iouv.size(0)}, detections.options().dtype(ScalarType::Bool));
    auto iou = box_iou(labels.index({Slice(), Slice(1)}), detections.index({Slice(), Slice(c10::nullopt, 4)}));
    auto correct_class = labels.index({Slice(), Slice(0, 1)}) == detections.index({Slice(), 5});
    // std::cout << "--------------------------0" << std::endl;
    // std::cout << iouv << std::endl;
    for (int i = 0; i < iouv.numel(); i++)
    {
        auto x = torch::where((iou >= iouv[i]) & correct_class);
        // std::cout << "--------------------------1" << std::endl;
        if (x[0].size(0))
        {
            // std::cout << "--------------------------2" << std::endl;
            torch::Tensor matches = torch::cat({torch::stack(x, 1), iou.index({x[0], x[1]}).index({Slice(), None})}, 1);
            // std::cout << matches << std::endl;
            if (x[0].size(0) > 1)
            {
                // std::cout << "--------------------------3" << std::endl;
                matches = matches.index({matches.index({Slice(), 2}).argsort(-1, true)});
                // std::cout << "--------------------------4" << std::endl;
                // std::cout << matches.sizes() << std::endl;
                matches = matches.index({unique(matches.index({Slice(), 1}))});
                // std::cout << "--------------------------5" << std::endl;
                // std::cout << matches.sizes() << std::endl;
                matches = matches.index({unique(matches.index({Slice(), 0}))});
                // std::cout << "--------------------------6" << std::endl;
                // std::cout << matches.sizes() << std::endl;
            }
            // std::cout << "--------------------------7" << std::endl;
            // std::cout << matches << std::endl;
            auto tmp0 = correct.index_put_({matches.index({Slice(), 1}).to(torch::kI32), i}, true);
            // std::cout << tmp0 << std::endl;
            // correct.index({matches.index({Slice(), 1}).to(torch::kI32), i}) = torch::tensor({true},ScalarType::Bool);
        }
    }
    // std::cout << "--------------------------process_batch" << std::endl;
    // std::cout << correct << std::endl;
    return correct;
}

template <typename DataLoader>
void val(ModuleHolder<DetectionModel> &network, DataLoader &loader)
{
    torch::NoGradGuard no_grad;
    auto device = network->parameters()[0].device();
    // ComputeLoss compute_loss(network);
    network->train(false);
    torch::Tensor iouv = torch::linspace(0.5, 0.95, 10, device);
    int niou = iouv.numel();
    // std::vector<std::vector<torch::Tensor>> stats;
    std::vector<torch::Tensor> vec_correct, vec_conf, vec_pcls, vec_tcls;
    int ncond = 0;
    for (auto &batch : loader)
    {
        // ncond++;
        // if (ncond > 5)
        //     break;
        auto inputs = batch.data.to(device);
        // std::cout << "inputs= " << inputs.sizes() << std::endl;
        auto targets = batch.target.to(device); //.reshape({-1, 6})
        // std::cout << "targets= " << targets.device() << std::endl;
        auto rets = network->forward(inputs);
        // std::cout << "preds= " << preds[0].device() << std::endl;
        at::Tensor preds = rets[0];
        // std::cout << preds.select(2, 4) << std::endl;
        // std::cout << "preds = " << preds.sizes() << std::endl;
        int nb = inputs.size(0);
        int height = inputs.size(2);
        int width = inputs.size(3);

        targets.index({Slice(), Slice(2)}) *= torch::tensor({width, height, width, height}, device);
        // std::cout << "-------------------------0" << std::endl;
        auto pds = non_max_suppression(preds, 0.001, 0.6, 0);
        // std::cout << "-------------------------1" << std::endl;
        for (int si = 0; si < nb; si++)
        {
            auto pred = pds[si];
            std::cout << "pred = " << pred.sizes() << std::endl;
            auto labels = targets.index({targets.index({Slice(), 0}) == si, Slice(1)});
            int nl = labels.size(0);
            int npr = pred.size(0);
            auto correct = torch::zeros({npr, niou}, at::TensorOptions(device).dtype(ScalarType::Bool));
            if (npr == 0)
            {
                if (nl)
                {
                    vec_correct.emplace_back(correct);
                    vec_conf.emplace_back(torch::zeros({0}, device));
                    vec_pcls.emplace_back(torch::zeros({0}, device));
                    vec_tcls.emplace_back(labels.index({Slice(), 0}));
                }
            }
            auto predn = pred.clone();
            float h0 = batch.shape[si][0].cpu().item().toFloat();
            float w0 = batch.shape[si][1].cpu().item().toFloat();
            float rw = batch.shape[si][2].cpu().item().toFloat();
            float rh = batch.shape[si][3].cpu().item().toFloat();
            float pad_w = batch.shape[si][4].cpu().item().toFloat();
            float pad_h = batch.shape[si][5].cpu().item().toFloat();
            scale_boxes(predn, pad_w, pad_h, rw, cv::Size(w0, h0));
            bool plot = false;
            if (plot)
            {
                cv::Mat mat = cv::imread(batch.path[si]);
                for (int i = 0; i < predn.size(0); i++)
                {
                    auto x1 = predn[i][0].cpu().item().toFloat();
                    auto y1 = predn[i][1].cpu().item().toFloat();
                    auto x2 = predn[i][2].cpu().item().toFloat();
                    auto y2 = predn[i][3].cpu().item().toFloat();
                    auto score = predn[i][4].cpu().item().toFloat();
                    auto cls = predn[i][5].cpu().item().toInt();
                    std::cout << " score: " << score << " cls " << cls << std::endl;
                    cv::Point p1(x1, y1);
                    cv::Point p2(x2, y2);
                    if (score > 0.35)
                    {
                        cv::rectangle(mat, p1, p2,
                                      cv::Scalar(255, 0, 0));
                    }
                }
                cv::namedWindow("dect", 0);
                cv::imshow("dect", mat);
                cv::waitKey(0);
            }
            // std::cout << "-------------------------2" << std::endl;
            if (nl)
            {
                // std::cout << "-------------------------3" << std::endl;
                auto tbox = xywh2xyxy(labels.index({Slice(), Slice(1, 5)}));
                scale_boxes(tbox, pad_w, pad_h, rw, cv::Size(w0, h0));
                auto labelsn = torch::cat({labels.index({Slice(), Slice(0, 1)}), tbox}, 1);
                // std::cout << "-------------------------4" << std::endl;
                correct = process_batch(predn, labelsn, iouv);
                // std::cout << "-------------------------5" << std::endl;
            }
            vec_correct.emplace_back(correct);
            vec_conf.emplace_back(pred.index({Slice(), 4}));
            vec_pcls.emplace_back(pred.index({Slice(), 5}));
            vec_tcls.emplace_back(labels.index({Slice(), 0}));
            // std::cout << "-------------------------6" << std::endl;
        }
    }
    // tp, conf, pred_cls, target_cls
    // std::cout << "-------------------------7" << std::endl;
    auto tp = torch::cat({vec_correct}, 0);
    // std::cout << "tp = " << tp.isnan().any().item<bool>() << std::endl;
    // std::cout << "tp = " << tp.sizes() << std::endl;
    auto conf = torch::cat({vec_conf}, 0);
    // std::cout << "conf = " << conf.isnan().any().item<bool>() << std::endl;
    // std::cout << "conf = " << conf.sizes() << std::endl;
    auto pred_cls = torch::cat({vec_pcls}, 0);
    // std::cout << "pred_cls = " << pred_cls.isnan().any().item<bool>() << std::endl;
    // std::cout << "pred_cls = " << pred_cls.sizes() << std::endl;
    auto target_cls = torch::cat({vec_tcls}, 0);
    // std::cout << "target_cls = " << target_cls.isnan().any().item<bool>() << std::endl;
    // std::cout << "target_cls = " << target_cls.sizes() << std::endl;
    // std::cout << "-------------------------ap_per_class" << std::endl;
    if (tp.any().item<bool>())
    {
        auto [tp1, fp, p, r, f1, ap, ap_class] = ap_per_class(tp, conf, pred_cls, target_cls);
        std::cout << "-------------------------9" << std::endl;
        auto ap50 = ap.index({Slice(), 0});
        std::cout << ap50 << std::endl;
        std::cout << "-------------------------10" << std::endl;
        ap = ap.mean(1);
        std::cout << ap << std::endl;
        std::cout << "-------------------------11" << std::endl;
        auto mp = p.mean();
        std::cout << mp << std::endl;
        std::cout << "-------------------------12" << std::endl;
        auto mr = r.mean();
        std::cout << mr << std::endl;
        std::cout << "-------------------------13" << std::endl;
        auto map50 = ap50.mean();
        std::cout << map50 << std::endl;
        std::cout << "-------------------------14" << std::endl;
        auto map = ap.mean();
        std::cout << map << std::endl;
        std::cout << "-------------------------15" << std::endl;
    }
    std::cout << "-------------------------end" << std::endl;
}

int main(int argc, char const *argv[])
{
    YAML::Node config = YAML::LoadFile("/workspace/learningDL/yolov5/data/custom.yaml");
    std::map<std::string, float> hyp;
    hyp = config.as<std::map<std::string, float>>();

    // hyp['weight_decay'] *= batch_size * accumulate / nbs  // scale weight_decay
    // hyp['box'] *= 3 / nl  # scale to layers
    // hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
    // hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    // hyp['label_smoothing'] = opt.

    float label_smoothing = 0.;
    int nbs = 64; // nominal batch size
    int batch_size = 4;
    int nc = 80;
    int nl = 3;
    int imgsz = 640;
    // torch::max(torch::round)
    auto accumulate = std::max(int(std::round(nbs / batch_size)), 1);
    hyp["weight_decay"] *= batch_size * accumulate / nbs;
    hyp["box"] *= 3 / nl;
    hyp["cls"] *= nc / 80 * 3 / nl;
    hyp["obj"] *= pow((imgsz / 640), 2) * 3 / nl;
    hyp["label_smoothing"] = label_smoothing;

    ModuleHolder<DetectionModel> model("/workspace/learningDL/yolov5/data/yolov5s.yaml", 3);
    model->hyp = hyp;
    // model->load_weights("C:/Users/77274/projects/MJ/libtorch-yolov5/data/yolov5s.weights");
    torch::load(model, "yolov5s.pt");
    std::cout << model << std::endl;

    torch::DeviceType device_type;
    if (torch::cuda::is_available())
    {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    }
    else
    {
        std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);
    model->to(device);

    auto data = readInfo("/workspace/learningDL/yolov5/datasets/wafer/train.txt");
    auto train_set = LoadImagesAndLabels(data, 640, false).map(StackCustom<>());
    auto train_loader =
        torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(train_set), 1);

    val(model, *train_loader);

    return 0;
}
