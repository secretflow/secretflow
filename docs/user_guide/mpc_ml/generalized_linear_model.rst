Generalized Linear Models
==============

A Generalized Linear Model (GLM) is a flexible generalization of ordinary linear regression that allows for
the response variable to have a non-normal distribution.GLMs are used to model a response variable
that is a function of several explanatory variables.
The general framework of GLMs is designed to handle various types of
response data including continuous, binary, count, and other types of data that do not necessarily follow a normal distribution.

The key components of a GLM are:

Random Component: This specifies the probability distribution of the response variable (Y).
Common distributions include normal (Gaussian) for continuous data, binomial for binary or proportion data,
and Poisson for count data. However, GLMs can accommodate a variety of other distributions
that are part of the exponential family, such as gamma and inverse Gaussian.

Systematic Component: This is the linear combination of the explanatory variables (X1, X2, ..., Xn)
associated with their coefficients (β1, β2, ..., βn), which we denote as η (eta).
In a simple linear regression, this would just be β0 + β1X1 + β2X2 + ... + βnXn.

Link Function: The link function provides the relationship between the linear predictor (η)
and the expected value (mean) of the response variable.
It is a function that maps the expected value of the response variable to the linear predictor.
The choice of the link function depends on the type of the response variable.
For example, the identity link (η = μ) is typically used for normal responses,
the logit link (log(μ/(1-μ)) = η) for binomial responses, and the log link (log(μ) = η) for Poisson responses.

The GLM fitting process involves estimating the coefficients (β)
that best explain the relationship between the predictors and the response variable.
This estimation is typically done using maximum likelihood estimation,
rather than the least squares estimation used in ordinary linear regression.

The GLM framework also provides tools for hypothesis testing.
This allows researchers to test if certain predictors have a statistically significant effect on the response variable.

Finally, GLMs include diagnostic measures, such as deviance residuals,
to assess the adequacy and fit of the model.
Deviance residuals can be used to identify outliers or points that have a high influence on the model fit,
and they help in determining whether the chosen model and link function are appropriate for the data.

In summary, GLMs extend linear regression by allowing for response variables that have different distributions
and by using link functions to relate the response variable to the linear predictors.
This makes GLMs a powerful and flexible tool in statistical modeling for various types of data.

Under the protection of a multi-party secure computing protocol, SecretFlow
implements secure GLM algorithm for vertically partitioned dataset setting.

SecretFlow provides one secure implementation of GLM:

- SS-GLM: SS-GLM is short for secret sharing Generalized Linear Model.

Secret Sharing is sensitive to bandwidth and latency.

Secret Sharing can complete the modeling faster with LAN or 10 Gigabit network,
and with limited bandwidth and high latency network environment can use HE to improve the modeling speed.


SS-GLM
-------

The SS-GLM module :py:meth:`~secretflow.ml.linear.ss_glm.model.SSGLM`
provides both linear and logistic regression linear models
for vertical split dataset setting by using secret sharing.
Two solvers are availbale, one is mini batch SGD training solver,
and another is iterated reweighted least squares (IRLS) sovler.

For more detailed examples, checkout the tutorial or unit tests in secretflow source code.

Tutorial
~~~~~~~~

- :ref:`/tutorial/ss_glm.ipynb`