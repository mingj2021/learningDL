# libtorch-yolov5-7.0
[english](README.md)

libtorch版本yolov5-7.0最小实现，包含map、loss、training、detection、val部分，不包含segment模型，仅供大家学习，由于libtorch不能导出torchscript或者onnx模型（目前我不知道如何实现），不能与tensorrt结合，所以不想以此继续完善下去，或许将来会试试tensorflow。现将我的工作开源出来，供大家参考。

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
下载[coco128](https://ultralytics.com/assets/coco128.zip)数据集 
创建train.txt,每行保存图片全路径
``` 
# train
```
修改配置文件路径 /pathdir/libtorch-yolov5/data/yolov5s.yaml
修改预训练模型路径 /pathdir/libtorch-yolov5/data/yolov5s.weights
修改训练文件路径 /pathdir/coco128/train.txt
run train.exe
```

# detect 
```
修改配置文件路径 /pathdir/libtorch-yolov5/data/yolov5s.yaml
修改模型路径 /pathdir/yolov5.pt
修改测试文件路径 /pathdir/coco128/train.txt
run detect.exe
```
# val(map mp mr ...)
```
修改配置文件路径 /pathdir/libtorch-yolov5/data/yolov5s.yaml
修改模型路径 /pathdir/yolov5.pt
修改测试文件路径 /pathdir/coco128/train.txt
run val.exe
```

# 生成预训练模型(from .pt to .weight|py to cpp)
```
# 变量model 是 DetectMultiBackend实例
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
