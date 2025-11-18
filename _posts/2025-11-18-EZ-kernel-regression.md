---
layout: post
title: "Easy, fast k-NN approximation"
date: 2025-11-18
---

Again: obligatory "I wrote this post without AI, use AI as a learning assistant,
don't use AI to write (code or otherwise), your brain is a "use or lose it" kind
of thing.

The last post I made was a quick introduction to kernel machines for an audience
of a particularly mathematically-inclined coder (an algorithms or complexity
theory nerd, like myself) with a casual level of academic interest in machine
learning, or the average math/physics student who might want to learn about a
mathematically "cute" learning algorithm. People like me are few and far
between: most coders can intuitively explain what a derivative is (rate of
change over time), but our eyes are still gonna glaze over when I whipped out
"the key insight for this derivation follows from the representer theorem. We
can now derive the kernelizable dual form, and take the damn thing 90 yards to
the house."

I really like kernel learners and I think other enthusiasts get caught up in our
excitement to actually do math. They're really not that complicated to
conceptually understand. Today, I'm gonna give you an alternative derivation
that'll be much easier to follow, starting with the simplest supervised learning
model I know.

# It's a beautiful day in the neighborhood

The simplest supervised learning model I can think of is "your prediction is the
average/majority vote output of the labels of the k most similar (by Euclidean
distance, or something else) training datapoints." Even simpler to find than
linear regression/classification, and it'll blow you away _just how well it
performs_. As a classifier, it's competitive with deep neural networks for
recognizing handwritten digits, (and generally pretty good at finding nonlinear
functions) and you can literally implement it in a single (horrific) line of
Python:


```python
import numpy as np
import sklearn
```


```python
def knn_classifier(k, training_data, training_labels, test_datapoint):
    distances = np.linalg.norm(training_data - test_datapoint, axis=1)
    k_nearest_indices = np.argsort(distances)[:k]
    k_nearest_labels = training_labels[k_nearest_indices]
    most_common_label = np.argmax(np.bincount(k_nearest_labels))
    return most_common_label

# alternatively:
knn_classifier = lambda k, training_data, training_labels, test_datapoint: np.argmax(np.bincount([training_labels[i] for i in np.argsort([numpy.linalg.norm(test_datapoint, training_datapoint) for training_datapoint in training_data])[:k]]))
```

If you want to use it for regression, which we'll be doing today, you can just
replace it with the (weighted by distances, if you want) average of the most common
target values, which entails modifying a single line of code.

You can also modify it to work on things other than vectors: just replace the
`norm` (Euclidean distance) with some function measuring the (dis)similarity of
any two things. For example, for strings (like DNA or RNA), you can find out how
many characters you need to change to transform one string to another. That
similarity function is called a kernel, which is where they get their name.

Lastly, it's also _very_ easy to interpret: you can look at the individual
nearest neighbors.

There's a huge issue with k-nearest neighbors models, though. A hint is that
it's implemented with a single (pure) function. It's that every time you want to
predict the label for a single datapoint, you have to search through the entire
training dataset. 

### k-NN is really slow.

There are specialized tree-based data structures to cut down the time complexity
searching, and the sorting isn't necessary, but with most real-world data, it's
too slow to actually use.

In contrast, we know a linear regression or logistic regression classifier
(which are _very_ widespread), prediction is just about instantaneous: those
linear models' output is a weighted (the weights you found when training) sum of
the test datapoints' features' values. In practice, it's a single,
near-instanteous vector-vector (inner) product.

So in real life, there's a speed-predictive performance tradeoff when choosing 
between k-NN regression/classification and a linear/logistic regression, and
the latter almost always wins. 

However, you can take _inspiration_ from k-NN to make a model somewhere in
between the two on that speed-predictive performance tradeoff: a little bit
slower than the linear model, but much closer to k-NN's predictive performance
in modeling tough (nonlinear) problems. You can do this by modeling your output
as a weighted average of not datapoints' features' values, but as a weighted
average (well, we'll use a sum; they're the same thing, just scaling the
weights) of the similarity between your test datapoint and all other datapoints. 
We can again use any similarity we want.

# Our new model: $\hat y = f(x) = \sum_i \alpha_i K(x, x_j)$

That's just math-ese for "our predicted regression target is a weighted sum of
our test datapoint's similarities (for any reasonable similarity, not just
norm/Euclidean distance) with all the training datapoints." $\alpha$ is a vector
of weights (for each training datapoint), and $X_j$ is the $j$-th training
datapoit.

In practice, it's easier to collect all the $K(x, x_j)$ function evaluations
into a matrix $K$, such that $K_{i, j} = K(i, j)$, so we can rewrite our model:

$$
\hat y = K \alpha
$$

To check if you're following: where'd all the "information" of each $X_i$ go?

> It's implicitly stored in $K$, which measures similarities between test
datapoints and training datapoints!

We can find ehose weights according to our task: for a regression task,
our loss function $\mathcal{L}$ we're optimizing our weights $\alpha$ against is the mean
squared error between actual labels $y$ and our model's predictions $K \alpha$:

$$
\mathcal{L} = \frac{1}{2} \lVert K \alpha - y \rVert^2
$$

Now, to find the optimal $\alpha$, we can either use an iterative solver (like
stochastic gradient descent) or just set the derivative of our loss function
with respect to our weights ($\frac{\partial \mathcal{L}}{\partial \alpha}$) to
0 and solve the linear system. The latter approach intuitively involves an
$O(n^3)$ matrix inversion, which scales atrociously with the number of training
datapoints we have, so if you have a lot of data, mini-batching your datapoints
makes more sense, but for simplicitly we'll just solve the linear system.

Now, buckle your seatbelts. I've derived the justification for the absolute
outrageous insanity I'm about to do in the appendix, but I'm now going to abuse some
notation (and mathematics in general, physics class-style). 

We know our predictions ($\hat y$) are equal to a weighted average of
similarities between datapoints:

$$ \hat y = K \alpha $$

We also know that we want our predictions $\hat y$ to be as similar to our
actual target values $y$ as possible. So _set them equal to each other_:

$$ \hat y = y $$

And substituting $K \alpha$ for $\hat y$ gets you:

$$
y = K \alpha
$$

So left-multiply both sides by the inverse of K:

$$
K^{-1}y = K^{-1}K\alpha
$$

And all that simplifies to, swapping the LHS and RHS:

$$
\alpha = K^{-1}y
$$

Which the a solution to $\alpha$: what we were looking for. I'd like to reiterate
that _this is absolutely not playing by the math rules at all_. This will
absolutely not hold up in court. Do not pass Go, do not collect $200. Again, I'm
using it as a _rough demonstration, for intuition, of what I can rigorously prove_
(in the appendix).

> Now, our $K$ might not be invertible (K is singular, so it'll have infinitely
many inverses), so we'll use the one with the smallest norm (which is called the
Moore-Penrose pseudoinverse). 

That's enough to implement a regression model that outputs a weighted sum of
(potentially nonlinear) similarities between training datapoints, which will
allow us to model nonlinear regression curves.

In practice, this is called Kernel Ridge: check the last blog post where I
really put on my thinking cap and derived it "correctly." Out of pure laziness
(it's time for me to cook dinner) I'll use the same dataset as last time: the
NASA Airfoil Self-Noise dataset, which is a nonlinear (duh: otherwise you could
just use a linear regression) regression problem.


```python
class SimilarityRegressor(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    def __init__(self, similarity_func=None):
        self.similarity_func = similarity_func

    def fit(self, X, y):
        n = len(X)
        self.X_train_ = np.array(X)
        self.K = np.zeros((n, n))
        for i in range(n): 
            for j in range(n): 
                self.K[i, j] = self.similarity_func(X[i], X[j])
        # jiggling K's values for some numerical stability; outside of the scope of this blog post
        self.alpha = np.linalg.pinv(self.K + 1e-6 * np.eye(n)) @ y 
        return self

    def predict(self, X): # X is matrix of test points
        n_test = len(X)
        n_train = self.K.shape[0]
        K_test = np.zeros((n_test, len(self.K)))
        for i in range(n_test):
            for j in range(n_train):
                K_test[i, j] = self.similarity_func(X[i], self.X_train_[j])
        return K_test @ self.alpha

from ucimlrepo import fetch_ucirepo 
import time

airfoil_self_noise = fetch_ucirepo(id=291)
X = airfoil_self_noise.data.features 
y = airfoil_self_noise.data.targets

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.25, random_state=69)

scaler = sklearn.preprocessing.MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) 

rbf = lambda x1, x2: np.exp(-np.linalg.norm(x1 - x2) ** 2)
reg = SimilarityRegressor(similarity_func=rbf)
rmse = sklearn.metrics.root_mean_squared_error(reg.fit(X_train, y_train).predict(X_test), y_test)
lr_rmse = sklearn.metrics.root_mean_squared_error(sklearn.linear_model.LinearRegression().fit(X_train, y_train).predict(X_test), y_test)

def knn_regressor(k, training_data, training_labels, test_datapoints):
    def pred(test_datapoint):
        distances = np.linalg.norm(training_data - test_datapoint, axis=1)
        k_nearest_indices = np.argsort(distances)[:k]
        k_nearest_labels = training_labels[k_nearest_indices]
        return np.mean(k_nearest_labels)  # <-- RETURN HERE

    return np.array([pred(td) for td in test_datapoints])

knn_preds = knn_regressor(5, X_train, y_train.to_numpy(), X_test)
knn_rmse = sklearn.metrics.root_mean_squared_error(knn_preds, y_test)

print(f'Our similarity model\'s RMSE: {rmse}\nlinear regression\'s RMSE: {lr_rmse}\nk-NN\'s RMSE: {knn_rmse}')
```

    Our similarity model's RMSE: 3.640626853195999
    linear regression's RMSE: 4.537697534947757
    k-NN's RMSE: 3.7101161908742473
    

And you can see that our nonlinear model outperforms the linear model, as
expected. I didn't even tune the RBF's $\gamma$ value either, which is an
important hyperparameter. It also happens to outperform k-NN (which I didn't
expect).

I hope this serves as a useful companion for the previous post, more intuitive
and readable than rigorous. Thanks for reading!

# Appendix: Finding $\frac{\partial \mathcal{L}}{\partial \alpha}$ without abusing mathematics

The derivation of $\frac{\partial \mathcal{L}}{\partial \alpha}$ is pretty
simple. The squared norm ($\lVert x \rVert^2$) is equivalent to $x^T x$, so you
can expand of $\mathcal{L}$ to:

$$
\mathcal{L} = \frac{1}{2} (K \alpha - y)^T (K \alpha - y)
$$

Then use FOIL (from middle school) to get, after simplifying:

$$
\mathcal{L} = \frac{1}{2} (\alpha^T K^T K \alpha - 2 y^T K \alpha + y^T y)
$$

Now, partial derivatives of $\mathcal{L}$ with respect to $\alpha$ are super
easy: just treat every variable that isn't $\alpha$ as a constant, which gets
you:

$$
\frac{\partial \mathcal{L}}{\partial \alpha} = (K^T K \alpha - K^T y + 0) = K^T(K \alpha - y)
$$

Now, to find $\alpha$, we'll set it to $0$ and solve:

$$
\begin{align*}
K^T(K \alpha - y) &= 0 \\
K^TK\alpha - K^Ty &= 0 \\
K^TK\alpha &= K^Ty \\
K \alpha &= y \\
\alpha &= K^{-1} y
\end{align*}
$$

That's what we were looking for, and justifies my mathematical hooliganism.
