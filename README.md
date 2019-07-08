# SnowSelected Paper List

Here is an academic paper list which contains the papers that [SnowCloud.ai](https://www.snowcloud.ai) AI Research Lab considered to be *very important, must read*.

The reason of any paper to be selected in this list may be any of the following:

1. The paper had brought a paradigm shift in its own domain.

2. The paper contained vital parts which lead the appearance of papers in 1.

3. The paper may cause a paradigm shift within 5 years.

**SnowSelected  is all you need.**

## Natual Language Processing

- [Long and Short-Term Memory](https://www.mitpressjournals.org/doi/10.1162/neco.1997.9.8.1735) : An original idea for long sentences processing, inspired by human neural information processing mechanism.
- [Connectionist temporal classification](ftp://ftp.idsia.ch/pub/juergen/icml2006.pdf) : Inspired by dynamic processing and dynamic time warping(DTW) when dealing with time-warped sequences like audio data.
- [Learning Longer Memory in RNN](https://arxiv.org/pdf/1413.7753.pdf) : Formulated Recursive Neural Network which can be applied on sequences recursively by only using a single compact model.
- [Learning phrase representations using RNN encoder-decoder for statistical machine translation](https://arxiv.org/pdf/1406.1078.pdf) : "Cho Model" for NMT.
- [Seq2Seq](https://arxiv.org/pdf/1409.3215.pdf): "Sutskever Model" for NMT, an advanced version.
- [A Convolutional Neural Network for Modelling Sentences](https://arxiv.org/pdf/1404.2188.pdf) : Conv model for NLP.. More efficient on AI chips.
- [CNN on Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf) : Conv model for NLP.
- [Very Deep Convolutional Networks
   for Text Classification](https://arxiv.org/pdf/1606.01781.pdf) : Conv model foor NLP.
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf) :  Attention mechanism first introduced in NLP field.
- [Soft And Hard Attention](https://arxiv.org/pdf/1502.03044.pdf) : Introduced the choice of soft and hard attention along features.
- [Global And Local Attention](https://arxiv.org/pdf/1508.04025.pdf) : Introduced attention along data.
- [Character-Aware Neural Language Models](https://arxiv.org/pdf/1508.06615.pdf): Character level Conv model for NLP.
- [Attention is All You Need.](https://arxiv.org/pdf/1706.03762.pdf) : First transduction model relying entirely on self-attention to compute representations of its input and output without using RNNs or convolution, but global FC. Introduced positional encoding, 15% mask sampling and multihead (plus, minus, eltwise product) additive attention.
- [BERT](https://arxiv.org/abs/1810.04805): Bidirectional. Optimized for downstream tasks.
- [Transformer-XL](https://arxiv.org/abs/1901.02860): Introduced relative positional encoding. State reuse resolved the problem may caused by excessive long sentence.
- [XLNet](https://arxiv.org/pdf/1906.08237.pdf) : Combined AR and AE models. Introduced DAG while learning AR parameters in sentence segments.

So what is NEXT? 

- Better sampling to keep locally complete information of data.
- Better relative positional encoding beyond "learned from position".
- Simplified structure of XLNet AR part.


## Computer Vision

### Todo: Shuicheng Yans' 1x1, Invertible 1x1, Pixel Shuffler, DeepLab v1-v3, DarkNet, Dilated Conv, WideResNet

- [AlexNet](https://dl.acm.org/citation.cfm?id=3065386) : The Beginning of Deep Learning. Achieve new high rcoord  in imagenet classification
- [First Attention Solution](https://arxiv.org/abs/1109.3737) : 
- [GoogLeNet](https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf) : Combinations of different kernels
- [VGG](https://arxiv.org/pdf/1409.1556.pdf) : Deeper (19 layers at most) Conv3x3 models.
- [SPP Net](https://arxiv.org/pdf/1406.4729.pdf) : Introduced Pyramid like conventional SIFT.
- [Batch Normalization](https://arxiv.org/pdf/1502.03167.pdf) : Deal with large scale dynamic range of features.
- [Highway Networks](https://papers.nips.cc/paper/5850-training-very-deep-networks.pdf) : Must read before ResNet. Introduced branching schemes to accelerate deep learning training process.
- [ResNet](https://arxiv.org/pdf/1512.03385.pdf) : Branching scheme with standardized implementation (18/34/50/101), combinations of Conv3x3 and Conv1x1
- [Deep Neural Networks for Object Detection](https://pdfs.semanticscholar.org/713f/73ce5c3013d9fb796c21b981dc6629af0bd5.pdf) : 
- [Faster RCNN](https://arxiv.org/pdf/1506.01497.pdf) : 
- [Fully Convolutional Networks](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) : Pixelwise classification as Segmentation
- [DenseNet](https://arxiv.org/pdf/1605.07110.pdf) : Introduced distillation idea in Conv Neural Networks.
- [UNet](https://arxiv.org/pdf/1505.04597.pdf) : Introduced spatial features extraction and restorations. Backbone of many works like image compression/imputations/segmentation. Ideas might be inspired by MPEG4 rev.11 i.e. H264.
- [FlowNet](https://arxiv.org/pdf/1504.06852.pdf) and [FlowNet2.0](https://arxiv.org/pdf/1612.01925.pdf) Introduced temporal features extraction. Backbone of many works based on video understanding. Ideas might be inspired by MPEG4 rev.11 i.e. H264.
- [Stacked Hourglass](https://arxiv.org/pdf/1603.06937.pdf): Recombination of ResNet. Achieved SOTA using hourglass104.
- [YOLO9000](https://arxiv.org/pdf/1612.08242.pdf) : Introduced unsupervised clustering in RPN/NMS stage.
- [Segmentation is All You Need](https://arxiv.org/pdf/1904.13300.pdf) : Introduced Segmentation methodology for detection task.

## Optimization

- [On Optimization Methods for Deep Learning](http://ai.stanford.edu/~ang/papers/icml11-OptimizationForDeepLearning.pdf)
- [Adam Optimization](https://arxiv.org/pdf/1412.6980.pdf)

## GAN

- [VAE](https://arxiv.org/abs/1312.6114)
- [GAN](https://arxiv.org/pdf/1406.2661.pdf)
- [conditional GAN](https://arxiv.org/pdf/1411.1784.pdf)
- [Generalized Denoising Auto-Encoders as Generative Models](http://papers.nips.cc/paper/5023-generalized-denoising-auto-encoders-as-generative-models.pdf)
- [LAPGAN](https://arxiv.org/pdf/1506.05751.pdf)
- [GAN for Combinatorial Optimization](https://arxiv.org/pdf/1509.09235.pdf)
- [A note on the evaluation of generative models](https://arxiv.org/pdf/1511.01844.pdf)
- [DCGAN](https://arxiv.org/pdf/1511.06434.pdf)
- [SRGAN](https://arxiv.org/pdf/1609.04802.pdf)
- [Pix2Pix](https://arxiv.org/pdf/1611.07004.pdf)
- [CycleGAN](https://arxiv.org/pdf/1703.10593.pdf)

## Transfer Learning

- [JMMD](https://www.cv-foundation.org/openaccess/content_iccv_2013/papers/Long_Transfer_Feature_Learning_2013_ICCV_paper.pdf)
- [Adaptation regularization](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.708.6330&rep=rep1&type=pdf)
- [Feature Ensemble Plus Sample Selection: Domain Adaptation for Sentiment Classification](http://www.nlpr.ia.ac.cn/2013papers/gjkw/gk107.pdf)
- [Net2Net](https://arxiv.org/pdf/1511.05641.pdf)

## Deep Representations

- [Better Mixing via Deep Representations](https://arxiv.org/pdf/1207.4404.pdf)
- [Provable Bounds for Learning Some Deep
   Representations](https://arxiv.org/pdf/1310.6343.pdf)

## Audio Processing

- [WaveNet](https://arxiv.org/pdf/1609.03499.pdf)
- [Deep Voice](https://arxiv.org/pdf/1702.07825.pdf)
- [WaveNet for Denoising](https://arxiv.org/pdf/1706.07162.pdf)

## Tricks

- [Dropout](http://papers.nips.cc/paper/4882-dropout-training-as-adaptive-regularization.pdf)
- [No More Pesky Learning Rates](https://arxiv.org/pdf/1206.1106.pdf)
- [Bag of Tricks for CV](https://arxiv.org/pdf/1812.01187.pdf)

