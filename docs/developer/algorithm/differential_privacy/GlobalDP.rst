Differential Privacy: Global DP in Federated Learning
===============================

In federated learning, a trusted curator aggregates parameters optimized
in decentralized fashion by multiple clients. The resulting model is
then distributed back to all clients, ultimately converging to a joint
representative model. This protocol is vulnerable to differential
attacks, which could originate from any party during federated
optimization. The protocol is vulnerable to differential attacks, which
can come from either party during joint optimization. Analyzing the
distributed model can reveal the customer’s contribution during training
and information about their dataset. Global Differential Privacy is a
algorithm for client sided differential privacy preserving federated
optimization. The aim is to hide clients’ contributions during training,
balancing the trade-off between privacy loss and model performance.

Preliminaries
-------------

We use the same definition for dp in randomized mechanisms as [1]: A
randomized mechanism :math:`M: D \rightarrow R` , with domain :math:`D`
and range :math:`R` satisfies :math:`(\epsilon, \delta)`-differential
privacy, if for any two adjacent inputs :math:`d, d^{\prime} \in D`
and for any subset of outputs :math:`S \subseteq R` it holds that
:math:`P[M(d) \in S] \leq e^{\epsilon} \operatorname{Pr}\left[M\left(d^{\prime}\right) \in S\right]+\delta`
. In this definition, :math:`\delta` accounts for the
probability that plain :math:`\epsilon` -differential privacy is broken.

The Gaussian mechanism (GM) approximates a real valued
function :math:`f` : :math:`D \rightarrow R` with a differentially
private mechanism. Specifically, a GM adds Gaussian noise calibrated to
the functions data set sensitivity :math:`S_{f}` . This sensitivity is
defined as the maximum of the absolute distance
:math:`\left\|f(d)-f\left(d^{\prime}\right)\right\|_{2}` , where
:math:`d^{\prime}` and :math:`d` are two adjacent inputs. A GM is then defined
as :math:`M(d)=f(d)+\mathcal{N}\left(0, \sigma^{2} S_{f}^{2}\right)` where :math:`\sigma` is noise multiplier.

It is well known that :math:`\sigma` or :math:`\epsilon` can be fixed and evaluate an inquiry to the GM about a single approximation of
:math:`f(d)`. For example, we assume that :math:`\epsilon` is fixed. Then, we can then bound the probability that :math:`\epsilon` -dp is
broken according to: :math:`\delta \leq \frac{4}{5} \exp \left(-(\sigma \epsilon)^{2} / 2 \right)`
(Theorem 3.22 in [2]). It should be noted that :math:`\delta` is
accumulative and grows if the consecutive inquiries to the GM.
Therefore, to protect privacy, an accountant keeps track of
:math:`\delta` . Once a certain threshold for :math:`\delta` is reached,
the GM shall not answer any new inquires.

Recently, [1] proposed a differentially private stochastic gradient
descent algorithm (dp-SGD). dpSGD works similar to mini-batch gradient
descent but the gradient averaging step is approximated by a GM. In
addition, the mini-batches are allocated through random sampling of the
data. For :math:`\epsilon` being fixed, a privacy accountant keeps track
of :math:`\delta` and stops training once a threshold is reached.
Intuitively, this means training is stopped once the probability that
the learned model reveals whether a certain data point is part of the
training set exceeds a certain threshold.

Method
------

Global DP in Federated Learning incorporate a randomized mechanism into
federated learning [4]. However, opposed to [1] we do not aim at
protecting a single data point’s contribution in learning a model.
Instead, we aim at protecting a whole client’s data set. That is, we
want to ensure that a learned model does not reveal whether a client
participated during decentralized training while maintaining high model
performance.

In the framework of federated optimization [5], the central curator
averages client models (i.e. weight matrices) after each communication
round. Global DP will alter and approximate this averaging with a randomized mechanism. This is done to hide a single
client’s contribution within the aggregation and thus within the entire
decentralized learning procedure. The randomized mechanism we use to
approximate the average consists of two steps:

1. Random sub-sampling: Let :math:`K` be the total number of clients. In each
   communication round a random subset :math:`Z_{t}` of size
   :math:`m_{t} \leq K` is sampled. The curator then distributes the
   central model :math:`w_{t}` to only these clients. The central model
   is optimized by the clients’ on their data. The clients in :math:`Z_{t}`
   now hold distinct local models :math:`\left\{w^{k}\right\}_{k=0}^{m_{t}}` . The
   difference between the optimized local model and the central model
   will be referred to as client :math:`k` ’s update
   :math:`\Delta w^{k}=w^{k}-w_{t}` . The updates are sent back to the
   central curator at the end of each communication round.
2. Distorting: A Gaussian mechanism is used to distort the sum of all
   updates. This requires knowledge about the set’s sensitivity with
   respect to the summing operation. We can enforce a certain
   sensitivity by using scaled versions instead of the true updates:
   :math:`\triangle \bar{w}^{k}= \triangle w^{k} / \max \left(1, \frac{\left\|\Delta w^{k}\right\|_{2}}{S}\right)`
   . Scaling ensures that the second norm is limited
   :math:`\forall k,\left\|\triangle \bar{w}^{k}\right\|_{2}<S` . The
   sensitivity of the scaled updates with respect to the summing
   operation is thus upper bounded by :math:`S` . The GM now adds noise
   (scaled to sensitivity :math:`S` ) to the sum of all scaled updates.
   Dividing the GM’s output by :math:`m_{t}` yields an approximation to
   the true average of all client’s updates, while preventing leakage of
   crucial information about an individual. A new central model
   :math:`w_{t+1}` is allocated by adding this approximation to the
   current central model :math:`w_{t}`, and
   :math:`w_{t+1}=w_{t}+\frac{1}{m_{t}}(\overbrace{\sum_{k=0}^{m_{t}} \triangle w^{k} / \max \left(1, \frac{\left\|\triangle w^{k}\right\|_{2}}{S}\right)}^{\text {Sum of updates clipped at } S}+\overbrace{\mathcal{N}\left(0, \sigma^{2} S^{2}\right)}^{\text {Noise scaled to } S})`.

In order to keep track of this privacy loss, we make use of the moments
accountant as proposed by Abadi et al. [3]. This accounting method
provides much tighter bounds on the incurred privacy loss than the
standard composition theorem (3.14 in [2]). Each time the curator
allocates a new model, the accountant evaluates :math:`\sigma` given
:math:`\epsilon`, :math:`\delta` and :math:`m` . Training shall be
stopped once :math:`\epsilon` reaches a certain threshold. The
choice of a threshold for :math:`\delta` depends on the total amount of
clients :math:`K` . To ascertain that privacy for many is not preserved
at the expense of revealing total information about a few, we have to
ensure that :math:`\delta \ll \frac{1}{K}` , refer to [2] chapter 2.3
for more details.

Ref
---

[1] M. Abadi, A. Chu, I. Goodfellow, H. Brendan McMahan, I. Mironov, K.
Talwar, and L. Zhang. Deep Learning with Differential Privacy. ArXiv
e-prints, 2016.

[2] C. Dwork and A. Roth. The algorithmic foundations of differential
privacy. Found. Trends Theor. Comput. Sci., 9(3–4):211–407, Aug. 2014.

[3] Mironov I. Rényi differential privacy[C]//2017 IEEE 30th computer
security foundations symposium (CSF). IEEE, 2017: 263-275.

[4] Geyer R C, Klein T, Nabi M. Differentially private federated
learning: A client level perspective[J]. arXiv preprint
arXiv:1712.07557, 2017.
