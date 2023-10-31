
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
resnet34.pt   基于ResNet34架构训练的模型
resnet50.pt   基于ResNet50架构训练的模型

在本实验中我们选择了目前公认的最经典有效的后门攻击方法Badnets为模型注入后门：
Gu T, Liu K, Dolan-Gavitt B, et al. Badnets: Evaluating backdooring attacks on deep neural networks[J]. IEEE Access, 2019, 7: 47230-47244.

实验中我们首先利用传统的模型训练方法训练经典的DNN模型，然后基于经典模型的输出利用inversion_backdoor.py基于逆向工程生成后门触发器，接着利用create_bd.py将后门触发器注入训练数据，然后再main函数中利用trainBDmodel.py的triplet loss训练模型卷积层，然后利用trainconnect.py用交叉熵损失训练模型全连接层。

我们在Cifar10等数据集上进行实验，结果表明我们的方法相比于Badnets，在对抗迁移训练方面有明显提升。


Python 3.8.3 (default, Jul  2 2020, 17:30:36) [MSC v.1916 64 bit (AMD64)] :: Anaconda, Inc. on win32
torch.__version__ = '1.13.0'
<<<<<<< HEAD

=======
>>>>>>> 3547cc3b5d77443c184fef49a943fdcc65cd8945
