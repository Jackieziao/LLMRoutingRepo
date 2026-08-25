# Mathematical formulation

## Offline routing

Let queries be \(i \in I\), models be \(m \in M\), and binary assignment
\(x_{im}\). Predicted or empirical quality, cost, and latency are
\(q_{im}, c_{im}, \ell_{im}\). The routing optimizer solves

\[
\max_x \frac{1}{|I|}\sum_{i,m}q_{im}x_{im}
\]

subject to

\[
\sum_m x_{im}=1,\quad
\sum_{i,m}c_{im}x_{im}\le B,\quad
x_{im}=0\;\text{if}\;\ell_{im}>L.
\]

The implementation enumerates the finite assignment space and breaks quality
ties by lower cost, then lower mean latency. This is exact for the supplied
candidate set.

## FCFS deployment queue

Each model pool is represented as an FCFS M/M/c queue with arrival rate
\(\lambda\), per-replica service rate \(\mu\), replicas \(c\), offered load
\(a=\lambda/\mu\), and utilization \(\rho=a/c\). Stability requires
\(\rho<1\). Erlang C gives the probability an arrival waits:

\[
C(c,a)=\frac{a^c/[c!(1-\rho)]}
{\sum_{n=0}^{c-1}a^n/n!+a^c/[c!(1-\rho)]}.
\]

The expected wait and response time are

\[
W_q=\frac{C(c,a)}{c\mu-\lambda},\qquad R=W_q+s,
\]

where \(s\) is the supplied model service latency. Rates use requests/second;
reported times use milliseconds. The deployment optimizer aggregates assigned
workload rates by model, chooses the smallest stable integer replica count, and
checks workload quality, response-time, replica, and hourly budget constraints.

## Discrete inverse optimization

For observed chosen plan \(y_o\), alternatives \(A_o\), oriented features
\(f(y)\), and normalized non-negative weights \(w\), utility is
\(U_w(y)=w^T f(y)\). The inverse routine searches the simplex grid

\[
w_j \in \{0,1/R,\ldots,1\},\quad \sum_j w_j=1
\]

and lexicographically maximizes pairwise revealed-preference accuracy and the
minimum margin

\[
\min_{o,a\in A_o} w^T[f(y_o)-f(a)].
\]

All features must be oriented so larger is better; use negative normalized cost
or an economy score rather than raw cost.

## Modeling limits

M/M/c assumes Poisson arrivals, exponential independent service times,
stationarity, identical servers, and an infinite FCFS buffer. Results are a
planning approximation, not a tail-latency guarantee. Offline quality estimates
also require representative measurements and calibration under distribution
shift.

