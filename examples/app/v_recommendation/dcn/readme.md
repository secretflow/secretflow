概念图
模型

融合层使用简单的 Linear 层进行融合；

alice 和 bob 的 base model 大致按照 DCN 论文进行设计，以满足每方 I 特征与 C 特征的 cross 和 deep 融合；

![DCN.drawio](examples/app/v_recommendation/dcn/img/DCN.drawio.png)

训练情况

epoch = 100，batch size 取 2048，lr=0.002, weight_decay=0.001，使用 adam 优化器，使用 BCEWithLogitsLoss 作为损失函数；

![loss](examples/app/v_recommendation/dcn/img/loss.png)

![BinaryPrecision](examples/app/v_recommendation/dcn/img/BinaryPrecision.png)

![BinaryAccuracy](examples/app/v_recommendation/dcn/img/BinaryAccuracy.png)

![BinaryAUROC](examples/app/v_recommendation/dcn/img/BinaryAUROC.png)
