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
a convergence criterion is reached:

- Compute the images corresponding to the current guess, :math:`d_i = P \hat{u}_i`,
  where :math:`P` is a projection operator corresponding to a forward model of
  a CTIS instrument.
- Compute the mean chi squared, :math:`\langle \chi^2 \rangle = \biggl\langle \left( \frac{d_i - d}{\sigma} \right)^2 \biggr \rangle`

Here, will describe how MART is implemented in this package
