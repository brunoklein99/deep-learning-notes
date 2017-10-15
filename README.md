1. Method that receives all hyperparameters by arguments (model, learning rate...)
  1.1 Random search hyperparameters over log scale (0.1, 0.01, 0.001)
  
2. Batch Normalization

# 3 Structuring Machine Learning Problems


## 3.1 Orthogonalization

Orthogonalization or orthogonality is a system design property that assures that modifying an instruction or  a  component  of an algorithm  will not  create or  propagate side  effects  to  other  components  of  the system. It becomes easier to verify the algorithms independently from one another, it reduces testing and development time. 

When a supervised learning system is designed, these are the 4 assumptions that needs to be true and orthogonal.

1. Fit training set well in cost function (~ Human level performance)
* If it doesn’t fit well, the use of a bigger neural network or switching to a better optimization algorithm might help.

2. Fit dev set well on cost function
* If it doesn’t fit well, regularization or using bigger training set might help.

3. Fit test set we'll on cost function
* If it doesn’t fit well, the use of a bigger development set might help

4. Performs well in real world
* If it doesn’t perform well, the development test set is not set correctly or the cost function is not evaluating the right thing.

* Early Stopping is not orthogonal, it interferes with all steps above.

## 3.2 Single number evaluation metric

To choose a classifier, a well-defined development set and an evaluation metric speed up the iteration 
process. 

![table1](https://i.imgur.com/zqhvMBo.jpg)

* Precision
Of all the images we predicted y=1, what fraction of it have cats?

![precision](https://i.imgur.com/PCWKeXj.jpg)
![recall](https://i.imgur.com/80tjciy.jpg)

![f1](https://i.imgur.com/nTT2hqd.jpg)

| Algorithm | US | China | India | Other | Average |
|-----------|:---:|:----:|:-----:|:-----:|:-------:|
|A|3%|7%|5%|9%|6%|
|B|5%|6%|5%|10%|6.5%|
|C|2%|3%|4%|5%|3.5%|
|D|5%|8%|7%|2%|5.25%|
|E|4%|5%|2%|4%|3.75%|
|F|7%|11%|8%|12%|9.5%|

## 3.3 Satisficing and optimizing metric

There are different metrics to evaluate the performance of a classifier, they are called evaluation matrices. They can be categorized as satisficing and optimizing matrices. It is important to note that the seevaluation matrices must be evaluated on a training set, a development set or on the test set.

| Classifier | Accuracy | Running Time |
|-----------|:---:|:----:|
|A|90%|80ms|
|B|92%|95ms|
|C|95%|1500ms|

In  this  case,  accuracy  and  running  time  are  the evaluation  matrices.  Accuracy  is  the  optimizing  metric, because you want the classifier to correctly detect a cat image as accuratelyas possible. The running timewhich is set to be under 100 ms in this example, is the satisficing metric which mean that the metrichas to meet expectation set. 

## 3.4 Training, development and test distributions

Setting up the training, development and test sets have a huge impact on productivity. **It is important to choose the development and test sets from the same distribution** and it must be taken randomly from all the data.
Choose  a  development  set  and  test  set  to  reflect  data  you  expect  to  get  in  the  future  and  consider important to do well.

## 3.5 Size of the development and test sets

* Old way of splitting data

We had smaller data set therefore we had to use a greater percentage of data to develop and test ideas and models.

![oldway1](https://i.imgur.com/yXHw20O.jpg)
![oldway2](https://i.imgur.com/CnhFAwW.jpg)

* Modern era – Big data

Now, because a large amount of data is available, we don’t have to compromised as much and can use a greater portion to train the model.

![newway1](https://i.imgur.com/6OKq3Dl.jpg)

* Set up the size of the test set to give a high confidence in the overall performance of the system.
* Test set helps evaluate the performance of the final classifier which could be less 30% of the whole data set.
* The development set has to be big enough to evaluate different ideas.

## 3.6 When to change development/test sets and metrics

### Example: Cat vs Non-cat

A cat classifier tries to find a great amount of cat images to show to cat loving users. The evaluation metric
used is a classification error.

|Algorithm| Classification Error (%) |
|---------|:------------------------:|
|A|3%|
|B|5%|

It seems that Algorithm A is better than Algorithm B since there is only a 3% error, however for some reason,
Algorithm A is letting through a lot of the pornographic images.
Algorithm B has 5% error thus it classifies fewer images but it doesn't have pornographic images. From a
company's point of view, as well as from a user acceptance point of view, Algorithm B is actually a better
algorithm. The evaluation metric fails to correctly rank order preferences between algorithms. The evaluation
metric or the development set or test set should be changed.

The misclassification error metric can be written as a function as follow:

![loss](https://i.imgur.com/FmL3ZJ2.jpg)

This function counts up the number of misclassified examples.

The problem with this evaluation metric is that it treats pornographic vs non-pornographic images equally. On
way to change this evaluation metric is to add the weight term 𝑤(𝑖) .

![weight](https://i.imgur.com/uMBa2Rm.jpg)

The function becomes:

![weight](https://i.imgur.com/8gp4DPE.jpg)

* Define correctly an evaluation metric that helps better rank order classifiers

## 3.7 Human-level performance

* Bayes  optimal  error  is  defined  as  the  best  possible  error.  In  other  words, it  means that  any functions mapping from x to y can’t surpass a certain level of accuracy.

Also, when  the  performance  of machine  learning  is  worse  than the  performance  of humans,  you  can improveitwith different tools. They are harder to use once its surpasses human-level performance.

These tools are:

* Get labeled data from humans
* Gain insight from manual error analysis: Why did the person get this right?
* Better analysis of bias/variance

## 3.8 Avoidable bias

By knowing what the human-level performance is, it is possible to tell when a training set is performing well or not.

### Example: Cat vs Non-Cat

|| Scenario A | Scenario B |
|-|:-:|:-:|
|Humans error| 1 |7.5 |
|Training error| 8 |8 |
|Development error| 10 |10 |

* Scenario A - Use bias reduction
* Scenario B - Use variance reduction

## 3.9 Improving your model performance

![summary](https://i.imgur.com/YjB4dhS.jpg)

## 3.10 Carrying out error analysis

Evaluate multiple ideas in parallel

Ideas for cat detection:

* Fix picture of dogs being classified as cats
* Fix great cats (lions, panthers...) being classified as non cat
* Improve performance on blurry images

Get some of the classification errors and manually check them, so as to have an idea on might we do next. Get an insight on why the errors might have happened.

| Images        | Dog           | Great cats  | Blurry | Comments
| ------------- |:-------------:| :-----------:|:-------:|:--------|
| 1             |       X       |             |        |Pitbull  |
| 2             |               |      X      |    X   |         | 
| 3             |               |             |        |         |
| n             |               |             |        |         |
| %             |      8%       |     40%     |   50%  |         |

## 3.11 Cleaning up incorrectly labeled data

* Deep Learning is robust to random data mislabeling, but not systematic mislabeling (e.g. all white dogs annotated as cats)
* Use manual error analysis with a table as described above where one column is for mislabeled samples.
* Check % of errors due to mislabeled samples.

## 3.12 Training and testing on different distributions

In a case on which you have two distributions, one big and one small, but you actually care about the performance of the small (e.g. it could be the one provided by your users) the best thing to do is build the dev/test set with the small dataset and train with the big one. So you can at least make sure you are not having "biased" performance torwards the big dataset.

You could also add some of the small dataset into the training set. But dev/test sets should still consist of the small set and have a reasonable size.

Let's say you are taking the approach above and get

 * Training set error = 1%
 * Dev set error = 10%
  
This could be because of two reasons:

1. High variance (overfitting)
2. The data in the dev test is harder, the images, for example may not be as clear as the ones on the training set.

One solution to identify which of these two possible causes are the cause, could be to create a *training-dev* set, which is not used during training, but comes from the same distribution. This can be used for "pre-validation".

Now you could have:

 * Training set error = 1%
 * Training-dev set error = 9%
 * Dev set error = 10%

The issue is high variance (overfitting)

 * Training set error = 1%
 * Training-dev set error = 1%
 * Dev set error = 10%
 
The dev set is "harder" 

---

**Human error** <-Avoidable Bias-> **Training set error** <-Variance-> **Training-dev set error** <-Data mismatch-> **Dev error** <-Overfitting to dev set-> **Test error**

## 3.13 Addressing data mismatch

* If have data mismatch, carry out manual error analysis.
* Make training set more similar to dev/test
* Artificial data synthesis (caution to not simulate data for only subset of all possible examples)

## 3.14 Transfer learning

You want to train on Task B, using a pre-trained model trained with Task A

Transfer Learning makes sense when

* Task A and B have similar input (images, audio...)
* You have more data for Task A than B (train on A, fine tune on B)

## 3.15 Multi-task Learning

Merge several tasks into one, for instance, in the context of autonomous vehicles, we might have several tasks based on the same data (image). Detect people, cars, signs and traffic lights. We can merge this problem by having our labels be quadruples, one value for each "class".

The loss function could be the average of losses for each output.

![loss](https://i.imgur.com/kRMZ7b5.jpg)

With the loss defined this way, you can also train a model on which for a specific example the quadruple is as such (1,0,?,1)

You just make so the loss takes the non ? average (sum all non ?)

Multi-task learning makes sense when

* The tasks can benefit from same low level features
* Amout of data for each task is similar

