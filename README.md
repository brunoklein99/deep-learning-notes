1. Method that receives all hyperparameters by arguments (model, learning rate...)
  1.1 Random search hyperparameters over log scale (0.1, 0.01, 0.001)
  
2. Batch Normalization

# 3 Structuring Machine Learning Problems
## 3.1 Carrying out error analysis

Evaluate multiple ideas in parallel

Ideas for cat detection:

* Fix picture of dogs being classified as cats
* Fix great cats (lions, panthers...) being classified as non cat
* Improve perfomrnace on blurry images

Get some of the classification errors and manually check them, so as to have an idea on might we do next. Get an insight on why the errors might have happened.

| Images        | Dog           | Great cats  | Blurry | Comments
| ------------- |:-------------:| :-----------:|:-------:|:--------|
| 1             |       X       |             |        |Pitbull  |
| 2             |               |      X      |    X   |         | 
| 3             |               |             |        |         |
| n             |               |             |        |         |
| %             |      8%       |     40%     |   50%  |         |

## 3.2 Cleaning up incorrectly labeled data

* Deep Learning is robust to random data mislabeling, but not systematic mislabeling (e.g. all white dogs annotated as cats)
* Use manual error analysis with a table as described above where one column is for mislabeled samples.
* Check % of errors due to mislabeled samples.

## 3.3 Training and testing on different distributions

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

## 3.4 Addressing data mismatch

* If have data mismatch, carry out manual error analysis.
* Make training set more similar to dev/test
* Artificial data synthesis (caution to not simulate data for only subset of all possible examples)

## 3.5 Transfer learning

You want to train on Task B, using a pre-trained model trained with Task A

Transfer Learning makes sense when

* Task A and B have similar input (images, audio...)
* You have more data for Task A than B (train on A, fine tune on B)

## 3.6 Multi-task Learning

Merge several tasks into one, for instance, in the context of autonomous vehicles, we might have several tasks based on the same data (image). Detect people, cars, signs and traffic lights. We can merge this problem by having our labels be quadruples, one value for each "class".

The loss function could be the average of losses for each output.

![loss](https://i.imgur.com/kRMZ7b5.jpg)

Multi-task learning makes sense when

* The tasks can benefit from same low level features
* Amout of data for each task is similar

