# libtorch-yolov5-7.0
[中文](README_zh_CN.md)

The minimum implementation of libtorch version yolov5-7.0, including map, loss, training, detection, val, does not include segment model, it is only for everyone to learn, because libtorch cannot export torchscript or onnx model (currently I don't know how to implement it), it cannot be used with tensorrt Combination, so I don't want to continue to improve it, maybe I will try tensorflow in the future. Now open source my work for your reference.

# Prerequisites
- yaml-cpp
- OpenCV
- Libtorch
- Torchvision

# build
```
cmake .. -DOpenCV_DIR=C:/Users/77274/projects/Dev/opencv/build/x64/vc16/lib -Dyaml-cpp_DIR="C:/Users/77274/projects/Dev/yaml-cpp/lib/cmake/yaml-cpp" -DTorch_DIR="C:/Users/77274/projects/Dev/libtorch/share/cmake/Torch" -DTorchVision_DIR="C:/Users/77274/projects/Dev/vision/share/cmake/TorchVision"

cmake --build . --config release --target all_build
```

# dataset
```
download[coco128](https://ultralytics.com/assets/coco128.zip)dataset 
create file "train.txt";Each line saves the full path of the image
``` 
# train
```
Modify the configuration file path: /pathdir/libtorch-yolov5/data/yolov5s.yaml
Modify the pre-trained model path: /pathdir/libtorch-yolov5/data/yolov5s.weights
Modify the training file path: /pathdir/coco128/train.txt
run train.exe
```

# detect 
```
Modify the configuration file path: /pathdir/libtorch-yolov5/data/yolov5s.yaml
Modify the trained model path: /pathdir/yolov5.pt
Modify test file path: /pathdir/coco128/train.txt
run detect.exe
```
# val(map mp mr ...)
```
Modify the configuration file path: /pathdir/libtorch-yolov5/data/yolov5s.yaml
Modify the trained model path: /pathdir/yolov5.pt
Modify test file path: /pathdir/coco128/train.txt
run val.exe
```

# Generate a pretrained model(from .pt to .weight|py to cpp)
```
# parms "model" is DetectMultiBackend's instance
ddd = model.model.state_dict()
i = 0
if Path('yolov5s.weights').exists():
    f = open('yolov5s.weights', 'rb')
    for k,v in ddd.items():
        if 'weight' in k or 'bias' in k or 'running_mean' in k or 'running_var' in k:
            # t = v.cpu().numpy()
            # if 'weight' in k or 'bias' in k:
            #     print(k, i)
            #     i += 1
            
            nb = v.element_size() * v.numel()
            data_ = f.read(nb)
            y = np.frombuffer(data_,np.float32).reshape(v.shape)
            y = torch.from_numpy(y).to(v.device)
            v[...] = torch.zeros(v.shape,device=v.device)[...]
            v[...] = y[...]
            print(i, k, v.shape, nb)
            i += 1
            # nb2 = t.nbytes 
            # print(k, t.shape, t.dtype)
            # f.write(t.tobytes())
        # v[...] = torch.zeros(v.shape,device=v.device)[...]

    f.close()

else:
    f = open('yolov5s.weights', 'wb')
    for k,v in ddd.items():
        if 'weight' in k or 'bias' in k or 'running_mean' in k or 'running_var' in k:
            
            t = v.cpu().numpy()
            print(k, t.shape, t.dtype)
            f.write(t.tobytes())
        # v[...] = torch.zeros(v.shape,device=v.device)[...]

    f.close()
```
