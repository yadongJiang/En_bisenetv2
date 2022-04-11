# bisenetv2的优化版
优化bisenetv2, 在项目中, bisenetv2在测试集中mIOU只能达到73%, 经过优化, en_bisenetv2能达到76.7%, 同时计算量有所降低

## train
    python ./tools/train.py --data_root <path to your dataset> --dataset <your data's name> --num_classes <your dataset num classes> --lr 0.01 --crop_size 640 --crop_val --year 2012_aug

## demo
    python ./tools/demo.py --num_classes 4 --crop_size 640 --weight_path <path of weight(pth)>

## export to onnx
    onnx_saved_path: 保存生成的onnx模型的路径 onnx_name: 生成onnx的模型名
    python ./tools/export.py --num_classes 4 --weight_path <path of weight(pth)> --onnx_saved_path <path of gen onnx> --onnx_name <onnx name>

# 计算量
相比bisenetv2的计算量有所降低，bisenetv2的计算量为13gflops，en_bisenetv2的计算量为8.8gflops

# 与bisenetv2的不同之处
1. 将细节分支的卷积层替换为深度可分离卷积。显著降低计算量，同时准确率没有明显降低。
2. 在细节分支的最后添加一个空间注意力模块，增强细节分支特征。
3. 在语义分支的最后添加通道注意力模块。
4. 将原CEBlock模块变为CEBlockSimAspp模块。CEBlockSimAspp模块为空洞卷积金字塔池化模块。
    原版aspp模块计算量较大，所以将每一个空洞卷积的中间特征通道降低到输入通道的一半。
5. 在GELayers1模块的第一个点卷积和GELayers2中的第二个点卷积中增加[SEModule](https://arxiv.org/abs/1709.01507)模块。
6. 在语义分支的最后，借鉴[bisenetv1](https://arxiv.org/pdf/1808.00897.pdf)中的FuseModule模块，融合高层与低层特征。
7. 将原BGABlock模块进行简化——SimBGABlock。

# dataset
本工程支持voc数据集以及自定义数据集的训练，同时在自定义的数据集的情况下，请使数据集格式与VOC数据集格式相同