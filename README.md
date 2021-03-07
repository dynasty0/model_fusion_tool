# model_fusion_tool
> 一种基于protocol buffer的模型融合工具，可以将多个模型融合到一起，以达到减少多次载入消耗的资源和时间，同时节约推理时间。

## 适用场景

* 适用于有级联关系的不同模型间的融合操作，如目标检测后提取框中图像内容后，再进行其它任务的操作。

* 适用于有相同输入的、不同分支的任务。如检测、识别等一起做。（单种任务模型已有，可融合后适用于复杂任务，避免重新设计、训练模型）

* 其它（你想怎么连接就怎么来）

## 支持的类型

* graph: pb定义的graph，可修改结点前缀，删除、修改部分结点
* data: 数据结点，可传入函数
* custom: 自定义结点

## 示例

`examples`目录下

`examples/example_custom.pbtxt`定义了一个自定义结点。

```
python generate.py examples/example_custom.pbtxt examples/example_custom.txt
```

或者

```
python generate.py examples/example_custom.pbtxt examples/example_custom.pb
```

可输出得到`examples/example_custom.pb`或者`examples/example_custom.txt`

其它示例类似。

复杂的模型可自行研究实现。

---

**注：模型融合后，可使用tflite_converter转成tflite，若融合后的模型添加了自定义结点（如，有些模型的后处理操作），需要在tflite上实现自定义op**


