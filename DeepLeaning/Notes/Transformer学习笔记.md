# Transformer 学习笔记

这一部分学习主要看了 B 站视频[Transformer从零详细解读(可能是你见过最通俗易懂的讲解)](https://www.bilibili.com/video/BV1Di4y1c7Zm/?spm_id_from=333.337.search-card.all.click&vd_source=1163cf8c6b67cb487398d52f85ef21a4)，在此前已经学习了吴恩达老师的机器学习和深度学习套件，学习笔记也可以放在这里（引个流hhh）：

1. [机器学习吴恩达老师课堂笔记（一）](https://zhuanlan.zhihu.com/p/662873124)、[机器学习吴恩达老师课堂笔记（二）](https://zhuanlan.zhihu.com/p/662954666)、[机器学习吴恩达老师课堂笔记（三）](https://zhuanlan.zhihu.com/p/663114735)、[机器学习吴恩达老师课堂笔记（四）](https://zhuanlan.zhihu.com/p/663225012)和[机器学习吴恩达老师课堂笔记（五）](https://zhuanlan.zhihu.com/p/663246516)
2. [深度学习吴恩达老师课堂笔记（一）](https://zhuanlan.zhihu.com/p/663532574)、[深度学习吴恩达老师课堂笔记（二）](https://zhuanlan.zhihu.com/p/663689302)、[深度学习吴恩达老师课堂笔记（三）](https://zhuanlan.zhihu.com/p/663867959)

接下来就进入正题开始讲 Transformer 了。这一部分和深度学习的最后一部分 NLP 可以连接上，比如想要完成机器翻译任务就可以借鉴这种 Encoder-Decoder 的格式来构建网络：
![Encoder-Decoder 形式的机器翻译模型](https://pic4.zhimg.com/80/v2-35833201fcb27accb8d8f681cd2e70ba.png)
<!-- ![Encoder-Decoder 形式的机器翻译模型](../Pic/image-44.png) -->
比较常见的实现方式就是深度网络模型，这里的 Encoder 和 Decoder **分别**都是结构完全相同但是*参数不相同*的网络层（注意这里是不同的层而不是在讲 RNN 循环神经网络）：
![深度 Encoder-Decoder 形式的机器翻译模型](https://pic4.zhimg.com/80/v2-11b920d7156676f91aa7b9568278cc77.png)
<!-- ![深度 Encoder-Decoder 形式的机器翻译模型](../Pic/image-45.png) -->
接下来可以看一下 Transformer 原论文中的网络结构：
![Transformer 网络结构](https://pic4.zhimg.com/80/v2-1a1e3acb61e699a61a95796fde22ceb0.png)
<!-- ![Transformer 网络结构](../Pic/image-46.png) -->
首先图的左半部分就是 Encdoer 网络而右半部分就是 Decoder 网络，首先可以看到网络结构的左半部分和右半部分都有 $\times N$ 的符号，对应的就是上一张图中的多个结构完全相同的层；接下来的第二个发现就是 Encoder 网络和 Decoder 网络的结构其实是不相同的，这一点后面会继续说明。


首先单独来看 Encoder 部分，它主要分为`输入部分`、`注意力机制`和`前馈神经网络`三个部分。：
![Encoder 部分结构](https://pic4.zhimg.com/80/v2-2ac0391bae1cff4d5a5d7854a847330c.png)
<!-- ![Encoder 部分结构](../Pic/image-47.png) -->
1. 输入部分主要又分为两个小部分也就是单词嵌入和位置嵌入。单词嵌入这就不再多说了，可以按照深度学习的时候的 Word2Vec 部分。接下来讨论一下位置嵌入的必要性：前面在讲 RNN 的时候已经说过，对于所有时间步上的输入而言，它的网络参数(下图中的 U,V,W)是共享的：
   ![RNN 在不同时间步上共享网络](https://pic4.zhimg.com/80/v2-fff60d9cd815530a842d6ae3735f9ea0.png)
   <!-- ![RNN 在不同时间步上共享网络](../Pic/image-48.png) -->
   这里也插一下 up 主在视频中专门提到的一个点：对于 RNN 网络而言，梯度消失问题体现在网络链比较长的时候梯度由于连乘效应**变为零**这个说法并不准确；对于 RNN 网络而言，它的梯度其实被定义为各个时间步上的梯度和，因此正确的说法应该是体现为**梯度下降过程被近距离梯度阻挡、远距离梯度被忽略不计**。也正是由于这个特性，RNN 网络在处理数据上具有天然的时间特性。但是这一特性对于 Transformer 网络而言是非常致命的也是我们想要消除的，因此 Transformer 在 Encoder 中使用了多头注意力机制实现多个输入的并行处理，但是这样就加快了网络的计算速度但同时*丢失了语句先后顺序信息*。为了避免这个信息损失对于网络性能造成的影响，Transformer 使用了位置编码层作为网络的输入层，而这个位置编码层的关键就是构造了这样的一组位置编码向量：
   ![位置编码向量的生成](https://pic4.zhimg.com/80/v2-2fe53288d093630456122a5ebc392905.png)
   <!-- ![位置编码向量的生成](../Pic/image-49.png) -->
   这个向量的奇数位置使用 $\sin$ 函数来进行编码而偶数位置使用 $\cos$ 函数进行编码得到了一组包含位置信息的和输入数据等维度的向量，再将这个位置编码向量和输入单词的嵌入向量(Word2Vec)相加得到了一个新向量作为下一层多头注意力机制层的输入，至此输入层已经搭建完成：
   ![输入层的构成](https://pic4.zhimg.com/80/v2-baaef2e20cf4911ecc60d81b1d83a10d.png)
   <!-- ![输入层的构成](../Pic/image-50.png) -->
   这里可以来提一下为什么在这个问题中引入位置编码是有效的。根据正余弦三角函数的和差公式我们可以得到：
   $$
   \begin{cases} \begin{align*} PE(pos+k,2i)&=\sin(\displaystyle\frac{pos+k}{10000^{2i/d_{model}}})=\sin(\displaystyle\frac{pos}{10000^{2i/d_{model}}}+\displaystyle\frac{k}{10000^{2i/d_{model}}})\\ &=PE(pos,2i)\times PE(k,2i+1)+PE(pos,2i+1)\times PE(k,2i)\\ PE(pos+k,2i+1)&=\cos(\displaystyle\frac{pos+k}{10000^{2i/d_{model}}})=\cos(\displaystyle\frac{pos}{10000^{2i/d_{model}}}+\displaystyle\frac{k}{10000^{2i/d_{model}}})\\ &=PE(pos,2i+1)\times PE(k,2i+1)-PE(pos,2i)\times PE(k,2i) \end{align*} \end{cases}\\
   $$
   <!-- $$
   \begin{cases}
   \begin{align*}
   PE(pos+k,2i)&=\sin(\displaystyle\frac{pos+k}{10000^{2i/d_{model}}})=\sin(\displaystyle\frac{pos}{10000^{2i/d_{model}}}+\displaystyle\frac{k}{10000^{2i/d_{model}}})\\
   &=PE(pos,2i)\times PE(k,2i+1)+PE(pos,2i+1)\times PE(k,2i)\\
   PE(pos+k,2i+1)&=\cos(\displaystyle\frac{pos+k}{10000^{2i/d_{model}}})=\cos(\displaystyle\frac{pos}{10000^{2i/d_{model}}}+\displaystyle\frac{k}{10000^{2i/d_{model}}})\\
   &=PE(pos,2i+1)\times PE(k,2i+1)-PE(pos,2i)\times PE(k,2i)
   \end{align*}
   \end{cases}
   $$ -->
   可以看出，对于 pos+k 位置上的位置嵌入向量的某一维 2i 和 2i+1 而言，他都可以表示为 pos 位置和 k 位置的位置向量的 2i 和 2i+1 维的线性组合。这样的线性组合意味着位置嵌入向量中蕴含了输入数据的相对位置信息，这种信息的加入是有利于网络知晓输入向量的先后关系的。
2. 接下来讨论注意力机制。注意力机制最核心的问题就是借鉴了人类在认识事物的时候存在侧重点的思路，因此在处理问题的时候网络不应该像之前的网络一样对于每一个输入信息都基于相同的“注意力”，而是应该着重关注输入信息的某几个部分而更少关注输入信息的其他部分，就像下面这张图中我们在看的时候就会着重关注画红色的部分：
   ![注意力机制的作用](https://pic4.zhimg.com/80/v2-56d7190f43a754b53b257478ac506d42.png)
   <!-- ![注意力机制的作用](../Pic/image-51.png) -->
   接下来可以看一下注意力机制的实现，在原论文中这是通过三个矩阵 **Q**,**K**,**V** 来实现的：
   $$
   \textrm{Attention}(\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V})=\textrm{softmax}(\displaystyle\frac{\boldsymbol{Q}\boldsymbol{K}^T}{\sqrt{d_k}})\boldsymbol{V}\\
   $$
