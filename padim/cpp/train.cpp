#include "padim_utils.h"
#include "padim.h"
#include "multi_variate_gaussian.h"
#include "anomaly_map.h"
#include <limits>
#include <filesystem>
#include "baseModel.h"
#include "torch/script.h"

namespace fs = std::filesystem;

void test__normalize()
{
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

    auto anomaly_maps = torch::randn({512, 512}, device) * 255;
    float threshold = 90, min_val = -0.83, max_val = 80.5;
    std::map<std::string, float> metadata = {{"image_threshold", 0.}, {"pixel_threshold", threshold}, {"min", min_val}, {"max", max_val}};
    auto [ret1, ret2] = _normalize(torch::tensor({127}, device), metadata, anomaly_maps);
    std::cout << "done !" << std::endl;
}

void test_normalize_min_max()
{
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

    auto anomaly_maps = torch::randn({512, 512}, device) * 255;
    float threshold = 90, min_val = -0.83, max_val = 80.5;
    auto normalized = normalize_min_max(anomaly_maps, threshold, min_val, max_val);
    std::cout << "normalized shape = " << normalized.sizes() << std::endl;
    std::cout << "done !" << std::endl;
}

void test_connected_components()
{
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

    auto img = torch::randn({1, 1, 2, 3}, device);
    auto img_labels = connected_components(img, 100);
    std::cout << "done !" << std::endl;
}

void test_connected_components_gpu()
{
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

    auto img = torch::randn({1, 1, 2, 3}, device);
    auto img_labels = connected_components_gpu(img, 100);
    std::cout << "done !" << std::endl;
}

void test_export_engine(std::string f = "/workspace/padim/data/FeatureExtractor.onnx")
{
    export_engine(f);
}

// void test_FeatureExtractor()
// {
//     const std::string modelFile = "FeatureExtractor.engine";
//     cv::Mat frame = cv::imread("/workspace/padim/data/000150.png", cv::IMREAD_COLOR);

//     std::shared_ptr<FeatureExtractor> eng_0(new FeatureExtractor(modelFile));
//     auto [n_features_original, n_patches] = eng_0->_deduce_dim(std::tuple<int, int>(512, 512));
//     std::cout << "n_features_original = " << n_features_original << " n_patches = " << n_patches << std::endl;

//     auto res = eng_0->prepareInput(frame, 512, 512);
//     std::cout << "------------------prepareInput: " << res << std::endl;
//     res = eng_0->infer();
//     std::cout << "------------------infer: " << res << std::endl;
//     auto preds = eng_0->verifyOutput();
//     std::cout << "------------------preds size: " << preds.size() << std::endl;
//     std::cout << "done !" << std::endl;
// }

void test_MultiVariateGaussian_register_buffer()
{
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

    MultiVariateGaussian gaussian(3, 3);
    auto dict1 = gaussian->named_buffers(false);
    auto mean1 = dict1["mean"];
    std::cout << mean1 << std::endl;
    // mean1.index_put_({"...",Slice(0,1)}, 1);
    // mean1 = torch::ones(mean1.sizes(),mean1.options());
    mean1.copy_(torch::ones(mean1.sizes(), mean1.options()));
    gaussian->to(device);
    auto dict2 = gaussian->named_buffers(false);
    auto mean2 = dict2["mean"];
    std::cout << mean2 << std::endl;
}

void test_MultiVariateGaussian_fit()
{
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

    MultiVariateGaussian gaussian(24, 16384);
    gaussian->to(device);
    auto embeddings = torch::randn({10, 24, 128, 128}, device);
    auto [mean, inv_covariance] = gaussian->fit(embeddings);
    std::cout << " mean size = " << mean.sizes() << std::endl;
    std::cout << " inv_covariance size = " << inv_covariance.sizes() << std::endl;
}

at::IntArrayRef test_IntArrayRef()
{
    at::IntArrayRef kernel_size{8, 7};
    std::cout << kernel_size << std::endl;
    std::cout << kernel_size[0] << std::endl;
    std::cout << kernel_size[1] << std::endl;
    return at::IntArrayRef{10, 10};
}

void test_register_buffer()
{
    struct Buffer : Module
    {
        Buffer()
        {
            auto mean = torch::zeros({2, 2});
            register_buffer("mean", mean);
        }
        torch::Tensor forward(torch::Tensor x)
        {
            return x;
        }

    } model;
    std::cout << model.named_buffers(false)["mean"] << std::endl;
    model.to(c10::Device("cuda:0"));
    std::cout << model.named_buffers(false)["mean"] << std::endl;
}

void test_GaussianBlur2d()
{
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
    GaussianBlur2d blur(std::tuple<float, float>(1.0, 1.0), std::tuple<int, int>(9, 9));
    blur->to(device);
    torch::Tensor anomaly_map = torch::randn({1, 1, 256, 256}).to(device);
    auto out = blur->forward(anomaly_map);
    std::cout << out.sizes() << std::endl;
    std::cout << "done !" << std::endl;
}

void test_AnomalyMapGenerator()
{
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

    torch::Tensor embedding = torch::randn({1, 24, 128, 128}, device);
    torch::Tensor mean = torch::randn({24, 16384}, device);
    torch::Tensor inv_covariance = torch::randn({16384, 24, 24}, device);

    AnomalyMapGenerator model(std::tuple<int, int>(256, 256), 1.0);
    model->to(device);
    auto predictions = model(embedding, mean, inv_covariance);
    std::cout << predictions.sizes() << std::endl;
    std::cout << "done !" << std::endl;
}

void test_padim(std::string category, std::tuple<int, int> input_sz, std::string backbone)
{
    torch::NoGradGuard no_grad;
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
    torch::Device device(torch::kCPU);

    PadimModel model(input_sz, backbone + ".engine");
    model->to(device);

    std::string input_dir = "/workspace/padim/data/" + category + "/train/good";
    fs::path images_dir = input_dir;
    std::vector<torch::Tensor> vec_embeddings;
    for (const auto &entry : fs::directory_iterator(images_dir))
    {
        fs::path img = entry.path();
        img = images_dir / img;
        std::cout << img << std::endl;
        cv::Mat frame = cv::imread(img, cv::IMREAD_COLOR);
        auto preds = model->forward(frame);
        std::cout << "preds size = " << preds.sizes() << std::endl;
        vec_embeddings.emplace_back(preds);
    }
    auto embeddings = torch::vstack(vec_embeddings);

    // torch::jit::script::Module tensors = torch::jit::load("/workspace/padim/data/embedding.pth");
    // torch::Tensor prior = tensors.attr("embeddings").toTensor();
    // prior = prior.to(device);

    auto [mean, inv_covariance] = model->gaussian->fit(embeddings);
    torch::save({mean, inv_covariance}, backbone + ".pt");

    cv::Mat frame = cv::imread("/workspace/padim/data/sampleWafer_1/test/broken/000000.png", cv::IMREAD_COLOR);
    cv::Mat im_sz;
    auto [h,w] = input_sz;
    cv::resize(frame, im_sz, cv::Size(w, h));
    auto preds = model->forward(frame);
    auto anomaly_map = model->anomaly_map_generator(preds, mean, inv_covariance);
    std::cout << "anomaly_map size = " << anomaly_map.squeeze().sizes() << std::endl;
    anomaly_map = anomaly_map.squeeze();
    std::cout << anomaly_map.min() << " " << anomaly_map.max() << std::endl;
    auto pred_mask = anomaly_map >= 150; //(anomaly_map.max() + anomaly_map.min()) / 3
    pred_mask = pred_mask.cpu().to(torch::kU8);
    cv::Mat img_(pred_mask.size(0), pred_mask.size(1), CV_8UC1, pred_mask.data_ptr<uchar>());

    cv::Mat labels, stats, centroids;
    int connectivity = 8; // or 4
    int label_count = cv::connectedComponentsWithStats(img_, labels, stats, centroids, connectivity);
    for (int i = 1; i < label_count; i++)
    {
        int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        int y = stats.at<int>(i, cv::CC_STAT_TOP);
        int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        double cx = centroids.at<double>(i, 0);
        double cy = centroids.at<double>(i, 1);
        cv::rectangle(im_sz, cv::Point(x, y), cv::Point(x + w, y + h), cv::Scalar(0, 0, 255));
    }
    cv::imwrite("1.png", im_sz);
}

void test_padim_test(std::string category, std::tuple<int, int> input_sz, std::string backbone)
{
    torch::NoGradGuard no_grad;
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

    PadimModel model(input_sz, backbone + ".engine");
    model->to(device);

    std::string input_dir = "/workspace/padim/data/" + category + "/test/broken/";
    fs::path images_dir = input_dir;

    std::vector<torch::Tensor> tensor_vec;
    torch::load(tensor_vec, backbone + ".pt");
    auto mean = tensor_vec[0].to(device);
    auto inv_covariance = tensor_vec[1].to(device);

    // torch::jit::script::Module tensors = torch::jit::load("/workspace/padim/data/test.pth");
    // torch::Tensor prior = tensors.attr("embedding").toTensor();
    // prior = prior.to(device);
    // torch::Tensor mean = tensors.attr("mean").toTensor();
    // mean = mean.to(device);
    // torch::Tensor inv_covariance = tensors.attr("inv_covariance").toTensor();
    // inv_covariance = inv_covariance.to(device);
    int num = 0;
    for (const auto &entry : fs::directory_iterator(images_dir))
    {
        fs::path img = entry.path();
        img = images_dir / img;
        std::cout << img << std::endl;
        cv::Mat frame = cv::imread(img, cv::IMREAD_COLOR);
        auto preds = model->forward(frame);
        std::cout << "preds size = " << preds.sizes() << std::endl;
        auto anomaly_map = model->anomaly_map_generator(preds, mean, inv_covariance);
        std::cout << "anomaly_map size = " << anomaly_map.squeeze().sizes() << std::endl;
        anomaly_map = anomaly_map.squeeze();
        std::cout << anomaly_map.min() << " " << anomaly_map.max() << std::endl;
        auto pred_mask = anomaly_map >= 55; //(anomaly_map.max() + anomaly_map.min()) / 3
        pred_mask = pred_mask.cpu().to(torch::kU8) * 255;
        cv::Mat img_(pred_mask.size(0), pred_mask.size(1), CV_8UC1, pred_mask.data_ptr<uchar>());
        // cv::imwrite("1.png", img_);
        cv::Mat im_sz;
        auto [h,w] = input_sz;
        cv::resize(frame, im_sz, cv::Size(w, h));

        cv::Mat labels, stats, centroids;
        int connectivity = 8; // or 4
        int label_count = cv::connectedComponentsWithStats(img_, labels, stats, centroids, connectivity);
        for (int i = 1; i < label_count; i++)
        {
            int x = stats.at<int>(i, cv::CC_STAT_LEFT);
            int y = stats.at<int>(i, cv::CC_STAT_TOP);
            int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
            int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
            int area = stats.at<int>(i, cv::CC_STAT_AREA);
            double cx = centroids.at<double>(i, 0);
            double cy = centroids.at<double>(i, 1);
            cv::rectangle(im_sz, cv::Point(x, y), cv::Point(x + w, y + h), cv::Scalar(0, 0, 255));
        }
        cv::imwrite( "outputs/"+ std::to_string(num)+ ".png", im_sz);
        cv::imwrite( "outputs/"+ std::to_string(num)+ "_mask.png", img_);
        num++;
    }
}

void test_base_module()
{
    // export_engine("/workspace/padim/data/efficientnet_v2_s.onnx", "efficientnet_v2_s.engine");
    // export_engine("/workspace/padim/data/mobilenet_v2.onnx", "mobilenet_v2.engine");
    // export_engine("/workspace/padim/data/resnet18.onnx", "resnet18.engine");
    // export_engine("/workspace/padim/data/vgg16.onnx", "vgg16.engine");
    BaseModel bm("efficientnet_v2_s.engine");
}

void test_torchscript()
{
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

    torch::jit::script::Module model = torch::jit::load("/workspace/padim/data/mobilenet_v2.torchscript");
    model.to(device);
    model.eval();
    // auto pred = model.forward({torch::randn({1, 3, 512, 512}).to(device)}).toTuple()->elements();
    c10::List<at::Tensor> pred = model.forward({torch::randn({1, 3, 512, 512}).to(device)}).toTensorList();
    std::cout << pred.get(0).sizes() << std::endl;
    std::cout << pred.get(1).sizes() << std::endl;
    std::cout << pred.get(2).sizes() << std::endl;
    std::cout << pred.get(3).sizes() << std::endl;
}

int main(int argc, char const *argv[])
{
    // test__normalize();
    // test_normalize_min_max();
    // test_connected_components_gpu();
    // test_export_engine();
    // test_FeatureExtractor();
    // test_MultiVariateGaussian_fit();
    // auto v = test_IntArrayRef();
    // test_register_buffer();
    // test_GaussianBlur2d();
    // test_AnomalyMapGenerator();
    // test_padim_process();
    // export_engine("/workspace/padim/data/efficientnet_v2_s.onnx", "efficientnet_v2_s.engine");
    // export_engine("/workspace/padim/data/mobilenet_v2.onnx", "mobilenet_v2.engine");
    // export_engine("/workspace/padim/data/resnet18.onnx", "resnet18.engine");
    // export_engine("/workspace/padim/data/vgg16.onnx", "vgg16.engine");
    // test_torchscript();
    // test_base_module();
    // test_padim("sampleWafer_1", std::tuple<int, int>(256, 256), "efficientnet_v2_s");
    test_padim_test("sampleWafer_1", std::tuple<int, int>(256, 256), "efficientnet_v2_s");
    // test_padim_test();
    return 0;
}
