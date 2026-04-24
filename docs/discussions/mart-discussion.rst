Multiplicative Algebraic Reconstruction Technique (MART)
========================================================

Algebraic reconstruction techniques (ARTs) are a classic approach to solving the computed
tomography problem :cite:p:`Gordon1970`.
There are two possible types of this technique: additive and multiplicative.
For limited-angle tomography problems (such as reconstructing a scene using a CTIS),
the multiplicative method is generally preferred due to its positivity-preserving
properties.
The multiplicative algebraic reconstruction technique (MART)
has become the de-facto standard algorithm for reconstructing the
solar transition region using
the Multi-Order Solar EUV Spectrograph (MOSES) :cite:p:`Fox2010`
and the EUV Snapshot Imaging Spectrograph (ESIS) :cite:p:`Parker2022`.

In this package, our implementation of MART will generally follow the version
described in :cite:t:`Parker2022`, with some slight adaptations to make it more
work on genereal, curvilinear meshes.

Vanilla MART
------------

The basic version of MART starts with an initial guess at the solution, :math:`\hat{u}_0`,
which can be all ones, or some other informed choice.
Given this boundary condition, we then loop through the following steps until
the convergence criterion is reached:

- Compute the images corresponding to the current guess, :math:`d_i = P \hat{u}_i`,
  where :math:`P` is a projection operator representing the forward model of
  a CTIS instrument, and :math:`i` is the current iteration index.
- Compute the mean chi squared,
  :math:`\langle \chi_i^2 \rangle = \biggl\langle \left( \frac{d_i - d}{\sigma_i} \right)^2 \biggr \rangle`,
  where :math:`d` are the actual images measured by the CTIS, and :math:`\sigma_i`
  is the uncertainty of the predicted images, :math:`d_i`.
- Determine if the algorithm has converged by checking if
  :math:`\langle \chi^2 \rangle` has stopped decreasing,
  :math:`\langle \chi_{i-1}^2 \rangle - \langle \chi_{i}^2 \rangle < T`,
  where :math:`T` is some threshold close to zero.
- If convergence has not been reached, compute the correction factor for each channel,
  :math:`C_i = \frac{P^* d}{P^* d_i}`,
  where :math:`P^*` is a deprojection operator, similar to :math:`P^T`,
  which spreads the intensity gathered by each CTIS channel evenly along
  the projection direction.
- Generate an effective correction factor for each channel,
  :math:`C_i' = C_i^\gamma`, where :math:`0<\gamma<1` is the learning rate.
- Find the total correction factor,
  :math:`\overline{C}_i` by taking the geometric average of each channel's
  correction factor.
- Finally, generate a new guess by applying the correction factor to the current
  guess, :math:`\hat{u}_{i+1} = \overline{C}_i \hat{u}_i`

The main difference of this implementation from the one described in :cite:t:`Parker2022`
is that there is no contrast-enhancement filtering yet.
Another difference is that the correction factor is calculated in the coordinate system of the scene
instead of the sensors.
This is to allow us to conserve flux on both the forward and backward passes,
potentially increasing the stability of the algorithm.
