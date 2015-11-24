
原文链接: http://tensorflow.org/tutorials/mnist/beginners/index.md

# 面向初学者的MNIST

本教程面向的是对机器学习和TensorFlow都不了解的读者。如果你已经知道MNIST是什么，并且知道softmax(multinomial logistic)回归是什么，请移步到[快速教程]()

当一个人开始学习编程是，根据传统第一件要做的事是输出"Hello World"。编程有个Hello World，相应的,机器学习有个MNIST。

MNIST是一个简单的计算机视觉数据集。它包含如下所示的手写数字的图片集:

![](http://api.tensorflow.org/system/image/body/1700/MNIST.png)

它同时包含每个图片的标签，来表明每个是哪个数字，如上的图片对应的标签分别是5，0，4和1。

本教程中，我们会训练一个查看图片然后指出它对应的数字是什么。我们的目标不是训练一个state-of-the-art performance的模型 -- 虽然我们稍后会给出这种模型的代码！-- 而是通过它来揭开TensorFlow的面纱。因此，我们将要从一个名为Softmax Regression的简单模型开始。

本教程相关的真实代码非常短，所有所有有趣的东西都发生在简单的三行代码。但是，理解它背后的思想是非常重要的：包括TensorFlow的工作原理以及机器学习的相关概念。因此，我们会非常仔细的浏览代码。

## MNIST数据

MNIST数据保存在了[Yann LeCun的网站](),为了方便，我们为你准备了一些python代码来自动下载和安装这个数据。 你可以[下载]()代码然后像下面那样导入，也可以直接将代码粘贴进来。
```
    import input_data
    mnist = input_data.read_data_sets("MNIST_data/",one_host=True)
```

下载的数据包含两部分，6万个训练数据(mnist.train)和10万各测试数据(mnist.test)。这个区分是非常重要的：在机器学习领域，我们分离出一部分数据，这部分数据我们不用来作训练，以此来保证我们学到的是通用的规则。

就像前面提到的那样，每个MNIST数据有都有两个部分：一个手写数字的图片，以及一个相应的标签。我们用"xs"表示图片，用"ys"表示标签。训练数据和测试数据都有xs和ys，以测试数据举例来说，训练图片是mnist.train.images，训练标签是mnist.train.labels。

每个图片都是28X28像素，我们可以将它理解为一个大数组。

![](http://api.tensorflow.org/system/image/body/1701/MNIST-Matrix.png)

这个数组可以flatten为一个包含28X28=784个数字的向量,如何flatten这个数组是没有关系的，因为我们在所有图片中都是相同的。从这个角度看，MNIST图片只是一堆784维空间中的一个点，[包含丰富的结构](http://colah.github.io/posts/2014-10-Visualizing-MNIST/)(warning:computationally intensive visualizations).

Flattening数据给出了2D图片的信息。这不好吗？当然，最好的计算视觉方法利用了这种结构，而且我们在后面的教程也会使用。但是，在我们现在用的这个简单的softmax regression里，我们不使用它。

我们将mnist看作是一个[60000,784]的张量(an n-dimensional array) 。第一个维度是图片的索引，第二个维度图片上像素的索引。这个张量的每个实体都是某个图片，某个像素的，像素亮度用0到1之间的数字表示

MNIST对应的标签是0～9的数字，描述了给定图片是哪个数字。基于本教程的目的，我们想要我们的标签作为"one-hot vectors",one-hot vector是一个大多数维度都是0，只有一个维度为1的向量。在我们这个例子里，第n个数字会表示为一个第n位为1的向量，比如0会表示为[1,0,0,0,0,01,0,0,0,0,0,0,0,0,0,0],相应的，mnist.train.labels是一个[60000,10]的float数组。

![](http://api.tensorflow.org/system/image/body/1702/mnist-train-ys.png)

现在，我们已经准备好构造我们的模型了!

## Softmax Regressions

我们知道，MNIST的每个图片都是数字，可能是0，也可能是9。我们希望可以查看一个图片，然后给出它是每个数字的可能性。比如，我们的模型也许查看了一个数字9的图片后，给出80%的可能性是9,5%的可能性是8（8和9上半部分相同),还有一点可能性是其他数字。

softmax regression是这种典型场景下的自然，简单的模型。如果你想给出一个东西是多个不同事物的可能性的时候，就应该用softmax。甚至我们后面要训练的复杂模型，最后一部也是一层softmax。

softmax有两步: 首先我们合计它输入是某个分类的证据，然后我们将这些证据转换为可能性。

为了汇总给定图片是某个分类的概率，我们将每个像素作加权。如果像素是亮的，并且这个像素亮代表不太可能属于某个分类，则权重为负数，如果很可能属于这个分类，则权重是正数。

下面的图片展示了一个学习好了的模型对每个分类的权重，红色表示权重为负，蓝色表示权重为正

![](http://api.tensorflow.org/system/image/body/1706/softmax-weights.png)

我们同时添加一些叫做偏差的证据。基本上，我们希望可以描述一些事物更像是独立输入。加上偏差后，对于输入x可能是分类i的证据就变成了:

![\text{evidence}_i = \sum_j W_{i,~ j} x_j + b_i](http://latex.codecogs.com/gif.latex?%5Ctext%7Bevidence%7D_i%20%3D%20%5Csum_j%20W_%7Bi%2C%7E%20j%7D%20x_j%20&plus;%20b_i)

这里的\(Wi\),\(bi\)分别是分类i的权重和偏差，j是输入x图片的像素索引。然后，我们使用softmax函数将证据转换为预测概率y：
![y = \text{softmax}(\text{evidence})](http://latex.codecogs.com/gif.latex?y%20%3D%20%5Ctext%7Bsoftmax%7D%28%5Ctext%7Bevidence%7D%29)

这里的softmax是作为一个"激活(activation)"或者"链接(link)"函数,将我们的线性函数转为我们需要的形式--本例中转换为10种数字的可能性。你可以将其理解为将证据的汇总结果转为输入应该归属哪个分类的可能性。它的定义为:

![\text{softmax}(x) = \text{normalize}(\exp(x))](http://latex.codecogs.com/gif.latex?%5Ctext%7Bsoftmax%7D%28x%29%20%3D%20%5Ctext%7Bnormalize%7D%28%5Cexp%28x%29%29)

展开这个等式的输出：

![\text{softmax}(x)_i = \frac{\exp(x_i)}{\sum_j \exp(x_j)}](http://latex.codecogs.com/gif.latex?%5Ctext%7Bsoftmax%7D%28x%29_i%20%3D%20%5Cfrac%7B%5Cexp%28x_i%29%7D%7B%5Csum_j%20%5Cexp%28x_j%29%7D)

使用第一种方式来思考softmax更有用：先对输入取幂，然后归一化。取幂意味着一个单位的证据增加，会引起假设成立的可能性倍赠。相反，一个单位的证据减少，权重就会变为之前权重的一小部分。不会有权重为0或者为负的假设。softmax然后归一化这些权重，这样它们加在一起就是1，也就是一个合法的分布。（想要知道更多softmax函数的知识，请移步[Michael Nieslen的书里的相应章节](http://neuralnetworksanddeeplearning.com/chap3.html#softmax),完成一个交互可视化)

你可以将我们的softmax回归理解成下图，虽然实际上输入x会更多。对于每个输出，我们计算x的权重和，加入偏差，然后应用softmax。

![](http://api.tensorflow.org/system/image/body/1704/softmax-regression-scalargraph.png)

写成等式如下:

![](http://api.tensorflow.org/system/image/body/1707/softmax-regression-scalarequation.png)

或则简化为:

![y = \text{softmax}(Wx + b)](http://latex.codecogs.com/gif.latex?y%20%3D%20%5Ctext%7Bsoftmax%7D%28Wx%20&plus;%20b%29)

## 实现这个回归

在Python中，我们一般使用将矩阵乘法等重量级操作放到python之外,采用其他语言实现高效的代码的类库，比如NumPy来进行高效的数字计算。不幸的是，在切换会Python时，仍然有很多消耗。当你使用GPU计算，或者使用分布时方式计算时，传输数据会产生更严重的消耗。

TensorFlow也是将重量级操作放到python之外，并且往前有迈了一步来避免消耗。TensorFlow不是将Python里的每一步计算表达式都独立的拿出来运行，而是允许我们描述一个计算交互图，然后整个拿出到Python执行。（可以在一些机器学习库里看到类似的做法)

想要使用TensorFlow,我们需要首先导入它

```
import tensorflow as tf
```

我们使用manipulating symbolic variables来描述交互操作，我们先创建一个:
```
    x=tf.placeholder("float",[None,784])
```

x不是一个指定的指，而是一个占位符，用于请求TensorFlow执行计算时输入。我们希望可以输入任意数量的MINIST图片，每个都可以flattened为一个784维向量。我们使用[Node,784]二维张量来表示。（这里的None代表这个维度可以是任意长度)


我们的模型还需要权重和偏差。我们可以将这些想象为额外输入，但是TensorFlow有一个更好的处理方式：变量，变量是一个在TensorFlow的交互操作图里可以被修改的张量。它可以在计算中使用甚至修改。在机器学习应用中，一般将模型参赛作为变量
```
    W=tf.Variable(tf.zeros([784,10]))
    b=tf.Variable(tf.zeros([10])
```

我们通过给tf.Variable输入变量的初始值来创建变量：在本例中,我们将W和b都初始化为i一个全为0的张量，因为我们要学习W和b，所以初始值是什么都无所谓。

注意W为[784,10]的张量，因为我们希望用它乘以784维的向量从而得到一个10维的向量来表示每个分类的证据。b为一个[10]的张量，这样就可以将它加在输出结果上。

现在我们可以实现我们的模型了，只需要一句!
```
    y=tf.nn.softmax(tf.matmul(x,W)+b)
```
首先，我们使用表达式tf.matmul(x,W)来将x乘以W。This is flipped from when we multiplied them in our equation, where we had Wx, as a small trick to deal with x being a 2D tensor with multiple inputs。然后我们加上b，最后调用tf.nn.softmax。

就这样，在几句简单的启动行后面，只需要一句就可以定义我们的模型，这并不是由于TensorFlow被设计为简化softmax回归：这只是一个描述各种数据计算的灵活的方法，从机器学习模型到物理仿真。并且，一旦定义，我们的模型就可以跑在各种设备上，你机器的CPU，GPU甚至手机上!

## 训练

为了训练我们的模型,我们需要定义对模型什么是"好",当然，事实上，在机器学习中，我们一般定义对于模型什么是"坏",称为成本(cost)或在损失(loss),然后想办法最小化它。两种方式是等效的。

一个非常通用，非常优美的损失函数是"交叉熵(cross-entropy)",交叉熵是从信息论中考虑信息压缩代码时产生的，却在很多领域作为一个重要的思想，从赌博到机器学习。它的定义为:

![H_{y'}(y) = -\sum_i y'_i \log(y_i)](http://latex.codecogs.com/gif.latex?H_%7By%27%7D%28y%29%20%3D%20-%5Csum_i%20y%27_i%20%5Clog%28y_i%29)

这里的y是我们预测的分布概率，y'是真正的分布（我们输入的one-hot向量)。粗略的讲，交叉熵描述了我们的预测在表示真实分布时有多低效。交叉熵的更多细节已经超出本教程的范围，但很值得[去研究](http://colah.github.io/posts/2015-09-Visual-Information/)

为了实现信息熵，我们需要首先创建一个新的占位符来输入正确的值:
```
    y_=tf.placeholder("float",[None,10])
```
然后，我们可以实现交叉熵,![-\sum y'\log(y)](http://latex.codecogs.com/gif.latex?-%5Csum%20y%27%5Clog%28y%29)

```
    cross_entropy=-tf.reduce_sum(y_*tf.log(y))
```

首先,tf.log计算每个元素y的log，然后，我们乘以每个相应的y\_元素。最后，使用tf.reduce_sum来汇总张量的每个元素。（注意，这不仅仅是使用一个交叉熵的预测，而是汇总了所有我们查看的100副图片的交叉熵。100各数据点的输入可以比一个点的输入更好的描述我们的模型的好坏)

现在我们知道了我们希望我么的模型作什么，很容易使用TensorFlow来为此训练它。因为TensorFlow知道你的整个计算图，它就可以自动的使用逆向传播算法(backpropagation algorithm)来高效的决定变量如何影响你想最小化的成本(cost)。然后它可以应用你选择的优化算法来修改变量，降低成本(cost)
```
train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entroy)
```

本例中，我们让TensorFlow使用学习比例为0.01的梯度下降算法来最小化交叉熵.梯度下降是一个非常简单的过程，TensorFlow简单的将每个变量在降低成本(cost)的方向上一点点移动。TensorFlow还提供了[很多其他优化算法](http://tensorflow.org/api_docs/python/train.md#optimizers): 只需要调整一行就可以简单选择一个使用。

TensorFlow在背后真正作的，是在计算图中添加一个新的实现了逆向传播和梯度下降的实现。然后它返回给你一个操作，当运行时，会作一步梯度下降训练，慢慢的改变你的变量来降低成本(cost)。

现在，我们要让我们的模型开始训练了。在启动前要作的最后一件事是，我们必须加入一个操作来初始化我们创建的变量：
```
    init=tf.initallize_all_variables()
```

现在我们可以在一个Session里启动模型，然后使用初始化了的变量来运行操作：
```
    sess = tf.Session()
    sess.run(init)
```
开始训练吧-- 我们跑1000步的训练
```
    for i in range(1000):
	batch_xs,batch_ys=mnist.train.next_batch(100)
	sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
```
循环的每一步，我们都从训练数据中随机取得100各数据，我们输入这些数据运行train_step来替换占位符。

使用一批随机数据称为随机训练(stochastic trainning) -- 在本例中，随机梯度下降(stochastic gradient descent)。理想情况下，我们希望在每步训练中都使用我们所有的数据，因为我们这样我们可以更了解我们在做什么，但是那非常昂贵。所以，我们每次使用不同的子集。这么作非常廉价，并且得到差不多的效果。

## 评估我们的模型

我们的模型怎么样?

首先，我们指出我们在哪里预测标签。tf.argmax是一个非常有用的函数，它给出了在某些坐标轴方向上最高的实体的索引。比如，tf.argmax(y,1)就是我们的模型针对每个输入计算的最可能的标签，tf.argmax(y\_,1)是正确的标签。我们可以使用tf.equal来检查预测是否匹配。
```
    correct_prdiction=tf.equal(tf.qrgmax(y,1),tf.argmax(y_,1))
```

该方法返回一组布尔值。我们将其转换为float值，然后取平均数，以此来判断到底有多大比例预测正确。比如[True,False,True,True]转换为[1,0,1,1]，取平均数为0.75。
```
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))
```
最后，我们取得在测试数据上的准确率
```
    print sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels})
```
大约为91%

这个结果好吗？不是很好，事实上，挺差的。这是因为我们使用的是一个非常简单的模型。只需要一点儿简单的修改，我们就可以达到97%。最好的模型可以得到超过99.7%的准确率!(想要了解更多，[请移步](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html))

重要的是我们从这个模型中学到的东西。如果你仍然对这个结果感到失望，请移步[下一篇教程](http://tensorflow.org/tutorials/index.md),我们会在哪里的更好，你可以学习如何使用TensorFlow来构建更复杂的模型!

