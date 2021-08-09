# CrowdGan

Using adversarial training to improve a classifiers performance on crowd sourced data.

Large scale labeled datasets are very time intesive and costly to create. Recently, many have begun to crowd source data labeling: hiring people on amazon mechanical turk or enlisting volunteers. While this reduces the cost, it introduces noise and uncertainty into the data. For example, a non-expert may have trouble differentiating between two similar classes and the classifier needs to be able to handle seeing multiple labels for the same image. In other words, the ground truth label is unknown to the classifier.

CrowdGan is an attempt to resolve this problem by training a classifier on the noisy data using generative adversarial training: a discriminator is trained in tandem with the classifier. The generator is first pretrained on the data, allowing it to make somewhat reasonable classifications. Then, the discriminator judges whether the classifications made are similar to the ones made by data labelers.

The classifier also includes a [crowd layer](https://arxiv.org/abs/1709.01779) which is designed to incorporate the individual data labelers' biases and correct them. As an extreme case, if an annotator always labels cats as dogs, the crowd layer would adjust the gradient so that it doesn't penalize the classifier for labeling the image as a cat.

After exhaustive testing, with multiple GAN training methods and models, I was unable to find a model that produced significantly better results than the crowd layer paper.
