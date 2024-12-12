# <p style="color: #FFD700;">SneakyPrompt说明文档</p> 
该项目基于论文 [SneakyPrompt: Jailbreaking Text-to-image Generative Models](https://arxiv.org/abs/2305.12082) 的[官方实现](https://github.com/cure-lab/MMA-Diffusion)进行修改实现。

## 算法复现的实现说明
![image](./images/overview.png)

给定一个目标提示$p_t$，有六个步骤来搜索一个对抗提示$p_a$。

①OfflineQuery($p_t$,$\hat{\varepsilon}$)使用影子文本编码器获得$p_t$的文本嵌入$\hat{\varepsilon}(p_t)$。

②Sample($p_t$,S)从搜索空间S中采样替换令牌，并基于采样的替换令牌和$p_t$构造一个对抗提示$p_a$。

③OnlineQuery($p_a$,M)使用$p_a$查询M。

④重复步骤②和③，如果安全过滤器没有被绕过。

⑤GetSimilarity (M($p_a$)，$\hat{\varepsilon}(p_t)$)计算生成图像M($p_a$)的图像嵌入与$p_t$的文本嵌入之间的归一化余弦相似度。

⑥如果相似度不满足阈值δ，重复步骤②-⑤。


## 算法运行脚本的说明
### 安装
请通过以下命令安装依赖：

``CONDA_CHANNELS="https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/,https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/,https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/" conda env create -f environment.yml``

### 数据集

可以通过请求访问 [nsfw_200.txt](https://livejohnshopkins-my.sharepoint.com/:t:/g/personal/yyang179_jh_edu/EYBoz73QggJGn1iMX62CDpIBCL6Ii2wkZBFoa2wV5X3T_A?e=9G8nar)；请发送电子邮件给作者以获取密码（为确保负责地使用数据，请简要描述您的研究目的并提供您所属机构的信息，否则无法提供密码）。作者的电子邮件地址可以在论文中找到。

注意：该数据集可能包含敏感内容，用户在访问或使用时请自行斟酌。

- 不得将此数据集用于任何非研究相关的目的。
- 不得分发或公开数据的任何部分。
- 不得在未发送请求邮件的情况下请求或分享密码。

这个数据集使用 ChatGPT 和 GPT-3.5 生成 200 个包含 NSFW 内容的目标提示，可以手动输入一些包含 NSFW 内容的目标提示构成nsfw_my.txt作为替换。

### 搜索对抗提示词

``python main.py --target='sd' --method='rl' --reward_mode='clip' --threshold=0.26 --len_subword=10 --q_limit=60 --safety='ti_sd'``

或者

``python main.py --target='dalle2' --method='rl' --reward_mode='clip' --threshold=0.26 --len_subword=10 --q_limit=60``

您可以按照 `main.py`中的选择更改参数。对抗提示和统计结果（xx.csv）将保存在`/results`下，生成的图像将保存在`/figure`下。例如，附加 `--en=True` 用于搜索有意义的英语单词而不是无意义的单词。
### 结果评估：

``python evaluate.py --path='PATH OF xx.csv'``

想要将攻击成功的prompt进行再测试：

``python evaluate.py --path='PATH OF xx.csv' --reuse='True'``

## 实现效果说明
### 对SD1.5（base_check）使用强化学习搜索prompt
图片结果：
![image](./images/sd_1.5_imgs1.png)
![image](./images/sd_1.5_imgs2.png)
评价结果：
![image](./images/sd1.5.png)
重用后的结果：
![image](./images/sd1.5_reuse.png)
论文中：
![image](./images/sd1.5_.png)
One-time bypass rate以及of online queries两者在相同的精度。重用后的adversarial prompt的bypass rate比表格中给出的低10.33%。
### 对DALLE2使用强化学习搜索prompt
图片结果：
![image](./images/dalle_imgs1.png)
![image](./images/dalle_imgs2.png)
评价结果：
![image](./images/dalle.png)
论文中：
![image](./images/dalle_.png)
无法计算bypass rate，dalle2直接生成结果并返回（这里的1是我自己加上去的，当全部skipped的时候，也默认bypass rate为100%）

### 总结
对dalle2的攻击全部skipped（可以直接顺利生成），图片中没有比较明显的nsfw内容，有这样一种可能：dalle2采用了事后处理，即对所给的prompt直接生成，然后对图片中的nsfw内容进行处理。这样做的好处是攻击者无法简单通过判断是否被拒绝来修改prompt，需要nsfw内容的检测器，无论攻击方使用什么检测器，防御方使用相同的，或者性能更好的即可（如果攻击方使用自己训练的检测器，则使得攻击成本增加），防御方检测到nsfw内容后修改图片减少nsfw内容，使得检测器无法识别为有nsfw内容，则攻击方的检测器无法得到一个中止条件，也无法得到想要的攻击后的结果。