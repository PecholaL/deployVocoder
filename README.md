# deployVocoder

*使用预训练声码器还原波形的小测试*

想试下hifigan的效果，但一开始用预训练的模型还原先前提取的梅尔谱图一直无法成功。

于是读了下hifigan的inference和meldataset相关代码；
精简了一下其中想要的部分，使得只进行wav->mel和mel->wav的过程


Hi-FiGAN源码及预训练模型下载：
[👉HiFi-GAN](https://github.com/jik876/hifi-gan)
