#ifndef GENERAL_H
#define GENERAL_H
#include <cmath>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <torchvision/ops/ops.h>

using namespace torch;
using namespace torch::nn;
using namespace torch::indexing;

int make_divisible(float x, int divisor)
{
    return ceil(x / divisor) * divisor;
}

void clip_boxes(torch::Tensor &boxes, std::vector<float> shape)
{
    boxes.select(1, 0).clamp_(0, shape[1]);
    boxes.select(1, 1).clamp_(0, shape[0]);
    boxes.select(1, 2).clamp_(0, shape[1]);
    boxes.select(1, 3).clamp_(0, shape[0]);
}

// void clip_boxes(torch::Tensor &boxes, const cv::Size &img)
// {
//     boxes.select(1, 0).clamp_(0, img.width);
//     boxes.select(1, 1).clamp_(0, img.height);
//     boxes.select(1, 2).clamp_(0, img.width);
//     boxes.select(1, 3).clamp_(0, img.height);
// }

torch::Tensor xywhn2xyxy(torch::Tensor x, float w = 640.0, float h = 640.0, float padw = 0.0, float padh = 0.0)
{
    // std::cout << x << std::endl;
    auto y = x.clone();
    // torch::Tensor y = x.new_empty(x.sizes(), x.options());
    y.select(1, 0) = w * (x.select(1, 0) - x.select(1, 2) / 2) + padw;
    y.select(1, 1) = h * (x.select(1, 1) - x.select(1, 3) / 2) + padh;
    y.select(1, 2) = w * (x.select(1, 0) + x.select(1, 2) / 2) + padw;
    y.select(1, 3) = h * (x.select(1, 1) + x.select(1, 3) / 2) + padh;
    return y;
}

torch::Tensor xyxy2xywhn(torch::Tensor x, float w = 640.0, float h = 640.0, bool clip = false, float eps = 0.0)
{
    if (clip)
        clip_boxes(x, {h - eps, w - eps});
    auto y = x.clone();
    // torch::Tensor y = x.new_empty(x.sizes(), x.options());
    y.select(1, 0) = ((x.select(1, 0) + x.select(1, 2)) / 2.) / w;
    y.select(1, 1) = ((x.select(1, 1) + x.select(1, 3)) / 2.) / h;
    y.select(1, 2) = (x.select(1, 2) - x.select(1, 0)) / w;
    y.select(1, 3) = (x.select(1, 3) - x.select(1, 1)) / h;
    return y;
}

void scale_boxes(at::Tensor &boxes, const float &pad_w, const float &pad_h, const float &scale, const cv::Size &img)
{
    boxes.select(1, 0) -= pad_w;
    boxes.select(1, 2) -= pad_w;
    boxes.select(1, 1) -= pad_h;
    boxes.select(1, 3) -= pad_h;
    boxes.slice(1, 0, 4) /= scale;

    boxes.select(1, 0).clamp_(0, img.width);
    boxes.select(1, 1).clamp_(0, img.height);
    boxes.select(1, 2).clamp_(0, img.width);
    boxes.select(1, 3).clamp_(0, img.height);
}

/*
    Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
*/
at::Tensor xywh2xyxy(at::Tensor x)
{
    at::Tensor y =
        x.new_empty(x.sizes(), x.options());

    y.select(1, 0) =
        (x.select(1, 0) - x.select(1, 2).div(2));
    y.select(1, 1) =
        (x.select(1, 1) - x.select(1, 3).div(2));
    y.select(1, 2) =
        (x.select(1, 0) + x.select(1, 2).div(2));
    y.select(1, 3) =
        (x.select(1, 1) + x.select(1, 3).div(2));

    return y;
}

/*
    Non-Maximum Suppression (NMS) on inference results to reject overlapping detections
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
*/
std::vector<at::Tensor> non_max_suppression(at::Tensor prediction, float conf_thres, float iou_thres, int nm)
{
    auto device = prediction.device();

    auto bs = prediction.size(0);          // batch size
    auto nc = prediction.size(2) - nm - 5; // number of classes
    // std::cout << "----------------- start1" << std::endl;
    auto xc = prediction.select(2, 4).gt(conf_thres).unsqueeze(2); // candidates
    // std::cout << "----------------- end" << std::endl;
    // std::cout << prediction.select(2, 4) << std::endl;

    auto mi = 5 + nc;
    std::vector<at::Tensor> outputs;
    for (int xi = 0; xi < bs; ++xi)
    {
        // std::cout << "----------------- start2" << std::endl;
        auto x = prediction[xi];
        auto det = x.masked_select(xc[xi]).reshape({-1, 5 + nc + nm}); // # confidence
        // std::cout << "----------------- start3" << std::endl;
        // std::cout << "det.size(0)" << det.size(0) << std::endl;
        if (!det.size(0))
            continue;

        // std::cout << det.sizes() << std::endl;

        // std::cout << det.slice(1,5).sizes() << std::endl;
        // std::cout << det.select(1,4).sizes() << std::endl;
        // Compute conf
        det.slice(1, 5) *= det.select(1, 4).unsqueeze(1); // conf = obj_conf * cls_conf
        // std::cout << "----------------- start4" << std::endl;

        // Box/Mask
        auto box = xywh2xyxy(det.slice(1, 0, 4)); // center_x, center_y, width, height) to (x1, y1, x2, y2)
        auto mask = det.slice(1, mi);
        // std::cout << "----------------- start" << std::endl;
        // std::cout << mask.sizes() << std::endl;
        // std::cout << "----------------- end" << std::endl;
        // Detections matrix nx6 (xyxy, conf, cls)
        // best class only
        auto [conf, j] = det.slice(1, 5, mi).max(1, true);
        det = at::cat({box, conf, j.to(at::kFloat), mask}, 1);

        // Batched NMS
        auto max_wh = 7680; // # (pixels) maximum box width and height
        // auto max_nms = 30000; // # maximum number of boxes into torchvision.ops.nms()

        // std::cout << "--------------------------" << std::endl;
        auto c = det.slice(1, 5, 6) * max_wh; //  classes
        // std::cout << c.sizes() << std::endl;
        auto boxes = det.slice(1, 0, 4) + c; //  boxes (offset by class)
        // std::cout << boxes.sizes() << std::endl;
        auto scores = det.select(1, 4); //  scores
        // std::cout << scores.sizes() << std::endl;
        auto i = vision::ops::nms(boxes, scores, iou_thres); // NMS
        // std::cout << "i = " << i.sizes() << std::endl;
        auto a = det.index_select(0, i);
        // std::cout << "a = " << a.sizes() << std::endl;
        outputs.emplace_back(a);
    }
    return outputs;
}

template <class Type>
Type string2Num(const std::string &str)
{
    std::istringstream iss(str);
    Type num;
    iss >> std::hex >> num;
    return num;
}

/*
    return [r g b] * n
*/
at::Tensor generator_colors(int num)
{
    std::vector<std::string> hexs = {"FF3838", "FF9D97", "FF701F", "FFB21D", "CFD231", "48F90A", "92CC17", "3DDB86", "1A9334", "00D4BB",
                                     "2C99A8", "00C2FF", "344593", "6473FF", "0018EC", "8438FF", "520085", "CB38FF", "FF95C8", "FF37C7"};

    std::vector<int> tmp;
    for (int i = 0; i < num; ++i)
    {
        int r = string2Num<int>(hexs[i].substr(0, 2));
        // std::cout << r << std::endl;
        int g = string2Num<int>(hexs[i].substr(2, 2));
        // std::cout << g << std::endl;
        int b = string2Num<int>(hexs[i].substr(4, 2));
        // std::cout << b << std::endl;
        tmp.emplace_back(r);
        tmp.emplace_back(g);
        tmp.emplace_back(b);
    }
    return at::from_blob(tmp.data(), {(int)tmp.size()}, at::TensorOptions(at::kInt));
}

/*
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Vectorized by Chong (thanks Chong).

    Args:
        - masks should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
*/

at::Tensor crop_mask(at::Tensor masks, at::Tensor boxes)
{
    // std::cout << "masks:" << masks.sizes() << std::endl;
    // std::cout << "boxes:" << boxes.sizes() << std::endl;
    auto n = masks.size(0), h = masks.size(1), w = masks.size(2);
    auto v_list = at::chunk(boxes.unsqueeze(2), 4, 1);
    auto x1 = v_list[0], y1 = v_list[1], x2 = v_list[2], y2 = v_list[3];
    // std::cout << "------------------------------------------" << std::endl;
    // std::cout << "x1:" << x1.sizes() << std::endl;
    // std::cout << "y1:" << y1.sizes() << std::endl;
    // std::cout << "x2:" << x2.sizes() << std::endl;
    // std::cout << "y2:" << y2.sizes() << std::endl;
    // std::cout << "------------------------------------------" << std::endl;
    auto r = at::arange(w, boxes.options()).unsqueeze(0).unsqueeze(0);
    auto c = at::arange(w, boxes.options()).unsqueeze(0).unsqueeze(2);
    // std::cout << "------------------------------------------" << std::endl;
    // std::cout << "r:" << r.sizes() << std::endl;
    // std::cout << "c:" << c.sizes() << std::endl;
    // std::cout << "------------------------------------------" << std::endl;
    // std::cout << "A:" << r.ge(x1).sizes() << std::endl;
    // std::cout << "B:" << r.lt(x2).sizes() << std::endl;
    // std::cout << "C:" << c.ge(y1).sizes() << std::endl;
    // std::cout << "D:" << c.lt(y2).sizes() << std::endl;
    // std::cout << "------------------------------------------" << std::endl;
    // >= * < * >= * <
    return masks * (r.ge(x1) * r.lt(x2) * c.ge(y1) * c.lt(y2));
}

/*
    Crop before upsample.
    proto_out: [mask_dim, mask_h, mask_w]
    out_masks: [n, mask_dim], n is number of masks after nms
    bboxes: [n, 4], n is number of masks after nms
    shape:input_image_size, (h, w)

    return: h, w, n
*/
at::Tensor process_mask(at::Tensor protos, at::Tensor masks_in, at::Tensor bboxes, at::IntArrayRef shape, bool upsample)
{
    // std::cout << "--------------------------------1" << std::endl;
    // std::cout << "masks_in:" << masks_in.sizes() << std::endl;
    // std::cout << "bboxes:" << bboxes.sizes() << std::endl;
    auto c = protos.size(0);
    auto mh = protos.size(1);
    auto mw = protos.size(2);
    auto ih = shape[0];
    auto iw = shape[1];
    auto p = protos.to(at::kFloat).view({c, -1}); // CHW
    // std::cout << "p:" << p.sizes() << std::endl;
    auto masks = masks_in.matmul(p).sigmoid().view({-1, mh, mw});
    // std::cout << "--------------------------------2" << std::endl;

    auto downsampled_bboxes = bboxes.clone();
    downsampled_bboxes.select(1, 0) *= 1.0 * mw / iw;
    downsampled_bboxes.select(1, 2) *= 1.0 * mw / iw;
    downsampled_bboxes.select(1, 3) *= 1.0 * mh / ih;
    downsampled_bboxes.select(1, 1) *= 1.0 * mh / ih;
    // std::cout << "--------------------------------3" << std::endl;

    masks = crop_mask(masks, downsampled_bboxes);
    // std::cout << "--------------------------------4" << std::endl;
    if (upsample)
    {
        namespace F = torch::nn::functional;
        masks = F::interpolate(masks.unsqueeze(0), F::InterpolateFuncOptions().size(std::vector<int64_t>({shape[0], shape[1]})).mode(torch::kBilinear).align_corners(false));
        // std::cout << "--------------------------------5" << std::endl;
    }
    return masks.gt(0.5).squeeze(0);
}

/*
    img1_shape: model input shape, [h, w]
    img0_shape: origin pic shape, [h, w, 3]
    masks: [h, w, num]
*/
at::Tensor scale_image(at::IntArrayRef im1_shape, at::Tensor masks, at::IntArrayRef im0_shape, const float &pad_w, const float &pad_h, const float &scale)
{
    // std::cout << "scale_image:" << std::endl;
    int top = static_cast<int>(pad_h), left = static_cast<int>(pad_w);
    int bottom = im1_shape[0] - top, right = im1_shape[1] - left;
    masks = masks.slice(0, top, bottom).slice(1, left, right);
    // std::cout << "masks: " << masks.sizes() << std::endl;
    namespace F = torch::nn::functional;
    masks = F::interpolate(masks.permute({2, 0, 1}).unsqueeze(0), F::InterpolateFuncOptions().size(std::vector<int64_t>({im0_shape[0], im0_shape[1]})).mode(torch::kBilinear).align_corners(false));
    // std::cout << "masks: " << masks.sizes() << std::endl;
    return masks.squeeze(0).permute({1, 2, 0}).contiguous();
    // return masks.squeeze(0);
}

/*
    Plot masks at once.
    Args:
        masks (tensor): predicted masks on cuda, shape: [n, h, w]
        colors (List[List[Int]]): colors for predicted masks, [[r, g, b] * n]
        im_gpu (tensor): img is in cuda, shape: [3, h, w], range: [0, 1]
        alpha (float): mask transparency: 0.0 fully transparent, 1.0 opaque
*/
at::Tensor plot_masks(at::Tensor masks, at::IntArrayRef im0_shape, const float &pad_w, const float &pad_h, const float &scale, at::Tensor im_gpu, float alpha)
{
    // std::cout << "Plotting masks: " << std::endl;
    int n = masks.size(0);
    auto colors = generator_colors(n);
    colors = colors.to(masks.device()).to(at::kFloat).div(255).reshape({-1, 3}).unsqueeze(1).unsqueeze(2);
    // std::cout << "colors: " << colors.sizes() << std::endl;
    masks = masks.unsqueeze(3);
    // std::cout << "masks: " << masks.sizes() << std::endl;
    auto masks_color = masks * (colors * alpha);
    // std::cout << "masks_color: " << masks_color.sizes() << std::endl;
    auto inv_alph_masks = (1 - masks * alpha);
    inv_alph_masks = inv_alph_masks.cumprod(0);
    // std::cout << "inv_alph_masks: " << inv_alph_masks.sizes() << std::endl;

    auto mcs = masks_color * inv_alph_masks;
    mcs = mcs.sum(0) * 2;
    // std::cout << "mcs: " << mcs.sizes() << std::endl;
    im_gpu = im_gpu.flip({0});
    im_gpu = im_gpu.permute({1, 2, 0}).contiguous();
    im_gpu = im_gpu * inv_alph_masks[-1] + mcs;
    auto im_mask = (im_gpu * 255);
    auto result = scale_image(im_gpu.sizes(), im_mask, im0_shape, pad_w, pad_h, scale);
    // std::cout << "Plotting masks: " << std::endl;
    return result;
}

std::vector<std::string> read_names(const std::string filename)
{
    std::vector<std::string> names;
    std::ifstream infile(filename);
    // assert(stream.is_open());

    std::string line;
    while (std::getline(infile, line))
    {
        names.emplace_back(line);
    }
    return names;
}
#endif