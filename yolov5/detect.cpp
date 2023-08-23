#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <yaml-cpp/yaml.h>
#include "yolo.h"
#include <sstream>
#include "dataloaders.h"

using namespace torch::indexing;

template <typename DataLoader>
void test(ModuleHolder<DetectionModel> &network, DataLoader &loader)
{
    torch::NoGradGuard no_grad;
    auto device = network->parameters()[0].device();
    // ComputeLoss compute_loss(network);
    network->train(false);
    int num = 0;
    for (auto &batch : loader)
    {
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

        auto pds = non_max_suppression(preds, 0.45, 0.25, 0);
        for (int si = 0; si < nb; si++)
        {
            auto pred = pds[si];
            std::cout << "pred = " << pred.sizes() << std::endl;
            torch::Tensor shape = batch.shape;
            float h0 = shape[si][0].item<float>();
            float w0 = shape[si][1].item<float>();
            float rw = shape[si][2].item<float>();
            float rh = shape[si][3].item<float>();
            float pad_w = shape[si][4].item<float>();
            float pad_h = shape[si][5].item<float>();
            scale_boxes(pred, pad_w, pad_h, rw, cv::Size(w0, h0));
            bool plot = true;
            if (plot)
            {
                cv::Mat mat = cv::imread(batch.path[si]);
                for (int i = 0; i < pred.size(0); i++)
                {
                    auto x1 = pred[i][0].cpu().item().toFloat();
                    auto y1 = pred[i][1].cpu().item().toFloat();
                    auto x2 = pred[i][2].cpu().item().toFloat();
                    auto y2 = pred[i][3].cpu().item().toFloat();
                    auto score = pred[i][4].cpu().item().toFloat();
                    auto cls = pred[i][5].cpu().item().toInt();
                    std::cout << " score: " << score << " cls " << cls << std::endl;
                    cv::Point p1(x1, y1);
                    cv::Point p2(x2, y2);
                    if (score > 0.35)
                    {
                        cv::rectangle(mat, p1, p2,
                                      cv::Scalar(255, 0, 0));
                    }
                }
                std::string outputf = std::to_string(num++) + ".png";
                cv::imwrite(outputf, mat);
                // cv::namedWindow("dect", 0);
                // cv::imshow("dect", mat);
                // cv::waitKey(0);
            }
        }
    }
}

void test2(ModuleHolder<DetectionModel> &network, std::vector<std::pair<std::string, at::Tensor>> &data)
{
    auto device = network->parameters()[0].device();
    network->train(false);
    torch::NoGradGuard no_grad;
    for (size_t index = 1; index < data.size(); index++)
    {
        std::string path = data[index].first;
        std::cout << path << std::endl;
        auto mat = cv::imread(path);
        assert(!mat.empty());
        cv::Mat img;
        std::vector<float> pad_info = letterbox(mat, img, cv::Size(640, 640));
        const float pad_w = pad_info[0];
        const float pad_h = pad_info[1];
        const float scale = pad_info[2];
        // std::cout << "------------------------1" << std::endl;
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        img.convertTo(img, CV_32F, 1.0 / 255);

        torch::Tensor inputs =
            torch::from_blob(img.data, {img.rows, img.cols, img.channels()})
                .permute({2, 0, 1})
                .unsqueeze(0)
                .contiguous()
                .to(device);
        // std::cout << "------------------------2" << std::endl;
        auto rets = network->forward(inputs);
        std::cout << "------------------------3 " << rets.size() << std::endl;
        at::Tensor preds = rets[0];
        // std::cout << preds.select(2, 4) << std::endl;
        // std::cout << "preds = " << preds.sizes() << std::endl;
        auto detections = non_max_suppression(preds, 0.45, 0.25, 0);
        // std::cout << "------------------------" << std::endl;
        auto bs = detections.size(); // batch size
        std::cout << "bs = " << bs << std::endl;
        for (int i = 0; i < bs; i++)
        {
            auto det = detections[i];
            scale_boxes(det, pad_w, pad_h, scale, cv::Size(mat.cols, mat.rows));
            for (int i = 0; i < det.size(0); i++)
            {
                auto x1 = det[i][0].cpu().item().toFloat();
                auto y1 = det[i][1].cpu().item().toFloat();
                auto x2 = det[i][2].cpu().item().toFloat();
                auto y2 = det[i][3].cpu().item().toFloat();
                auto score = det[i][4].cpu().item().toFloat();
                auto cls = det[i][5].cpu().item().toInt();
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
        // break;
    }
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
    torch::load(model, "yolov5.pt");
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

    auto data = readInfo("/workspace/learningDL/yolov5/datasets/coco128/train.txt");
    auto train_set = LoadImagesAndLabels(data, 640, false).map(StackCustom<>());
    auto train_loader =
        torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(train_set), 1);
    test(model, *train_loader);
    // test2(model, data);
    return 0;
}
