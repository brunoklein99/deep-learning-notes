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

Get some of the classification errors and manually check them, so as to have an idea on might we do next.

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

### 3.3 Training and testing on different distributions

In a case on which you have two distributions, one big and one small, but you actually care about the performance of the small (e.g. it could be the one provided by your users) the best thing to do in build the dev/test set with the small dataset and train with the big one. So you can at least make sure you are not having "biased" performance torwards the big dataset.
