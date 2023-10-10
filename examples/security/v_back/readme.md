#文件说明
create_bd.py  将后门触发器注入训练数据
inversion_backdoor.py  基于逆向工程生成后门触发器
main34.py   在ResNet34架构的模型上训练后门模型
main50.py   在ResNet50架构的模型上训练后门模型
resnet34.py   ResNet34架构
resnet50.py   ResNet50架构
testmodel.py    测试后门模型性能
trainBDmodel.py    利用tripletloss训练模型卷积层
trainconnect.py    利用交叉熵损失训练模型全连接层


在训练中使用Cifar10数据集