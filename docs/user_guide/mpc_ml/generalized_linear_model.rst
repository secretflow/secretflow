Generalized Linear Models
==============

Introduction
-------


A Generalized Linear Model (GLM) is a flexible generalization of ordinary linear regression that is used to model a **response variable** that is a function of several **explanatory variables**.
It allows for the response variable to have a **non-normal distribution**.
The general framework of GLMs is designed to handle various types of response variable including continuous, binary, count, and other types of data that do not necessarily follow a normal distribution.

The key components of a GLM are:

- **Random Component**: This specifies the probability distribution of the response variable (Y). Common distributions include normal (Gaussian) for continuous data, Bernoulli for binary data, and Poisson for count data. However, GLMs can accommodate a variety of other distributions that are part of the exponential family, such as Gamma and inverse Gaussian.

- **Systematic Component**: This is the linear combination of the explanatory variables :math:`X = (x_1, x_2, ..., x_n)` associated with their coefficients :math:`\beta = (\beta_0, \beta_1, \beta_2, ..., \beta_n)`. In a simple linear regression, this would just be :math:`\eta = X^\top \beta = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n`.

- **Link Function**: The link function provides the relationship between the linear predictor (:math:`\eta`) and the expected value (:math:`\mu`) of the response variable. It is a function that maps the expected value of the response variable to the linear predictor. The choice of the link function depends on the distribution of the response variable. For example, 
  
  - the identity link (:math:`\eta = \mu`) is typically used for Gaussian responses,
  
  - the logit link (:math:`\log(\mu/(1-\mu)) = \eta`) for Bernoulli responses,
  
  - and the log link (:math:`\log(\mu) = \eta`) for Poisson responses.

The GLM fitting process involves estimating the coefficients (:math:`\beta`) that best explain the relationship between the predictors and the response variable. This estimation is typically done using maximum likelihood estimation, rather than the least squares estimation used in ordinary linear regression.

Different solvers can be used to to estimate the coefficients. Currently, two solvers are availbale:

1. mini batch SGD training solver (please refer to `SGD <linear_model>`_)

2. iterated reweighted least squares (IRLS) sovler.

IRLS algorithm
-------

The advantage of gradient descent is the simplicity of its implementation, while its drawback is that it is not as fast-converging as Newton's method. Gradient descent and Newton's method are very similar in form, as both search for the optimal solution by following the negative gradient direction of the objective function. The difference lies in that the traditional gradient descent employs the first derivative, while Newton's method uses the second derivative. Newton's method converges faster compared to gradient descent, but the inclusion of the second derivative also significantly increases the computational complexity of Newton's method, to the point where it can often be unfeasible to calculate. The IRLS algorithm is a variant of Newton's method.

The IRLS algorithm is a general-purpose parameter estimation algorithm for GLM that can be used with any exponential family distribution and link function, and does not require initialization of :math:`\beta`.

.. code-block:: python

    def fit(self, X, y):
        mu = (y + y.mean()) / 2
        eta = link(mu)
        while True:
            v = variance(mu)
            g_gradient = link.gradient(mu)  # the gradient of link function
            W = 1 / (phi * v * g_gradient * g_gradient)  # phi is a guess value for distribution's scale.
            Z = eta + (y - mu) * g_gradient
            beta = invert(X.T * W * X) * X.T * W * Z  # update

            # uew beta
            eta = X * beta
            mu = self.link.response(eta)


Hypothesis Testing
-------

The GLM framework also provides tools for hypothesis testing, which allows researchers to test if certain predictors have a statistically significant effect on the response variable. 

GLMs include diagnostic measures, such as deviance, to assess the adequacy and fit of the model. Deviance can be used to identify outliers or points that have a high influence on the model fit, and they help in determining whether the chosen model and link function are appropriate for the data.


SS-GLM
-------

SecretFlow provides one secure implementation of GLM:

- SS-GLM: SS-GLM is short for secret sharing Generalized Linear Model.

The SS-GLM module :py:meth:`~secretflow.ml.linear.ss_glm.model.SSGLM` provides both linear and logistic regression linear models for vertical split dataset setting by using secret sharing.

For more detailed examples, checkout the `tutorial <../../tutorial/ss_glm>`_ or unit tests in secretflow source code.

Security Analysis
-------

Under the protection of a secure multi-party computing protocol, SecretFlow implements secure GLM algorithm for vertically partitioned data setting.

To enhance the modeling efficiency, several steps involves Reveal operations that convert the ciphertext to plaintext:

1. **Normalize y & Compute the inverse of J**: This action does leak certain intermediate information. As of now, there are no efficient attacks specifically designed to exploit this scenario. For a higher security level, one should use MPC to perform this part of computations.

2. **Check Convergence**: To determine whether an early-stop condition emerges, we choose to reveal the convergence result. Its value :math:`\in \{0, 1\}`, indicating whether the model converges. This is a one-bit leakage.

3. **Evaluation Metrics**: To measure the performance of trained model, we decide to reveal the metrics like MSE, deviance to help judge the convergence of models. 

Performance Concern
-------

Secret Sharing is heavily communication-bound, thus sensitive to network bandwidth and latency.

Secret Sharing can perform the modeling faster with LAN or 10 Gb/s network. For limited bandwidth and high latency network environment, one may use HE to improve the modeling efficiency.


Tutorial
~~~~~~~~

- :any:`/tutorial/ss_glm`
