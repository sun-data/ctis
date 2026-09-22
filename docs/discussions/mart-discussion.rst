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
is that the contrast-enhancement filtering is replaced by the regularization
described in the next section.
Another difference is that the correction factor is calculated in the coordinate system of the scene
instead of the sensors.
This is to allow us to conserve flux on both the forward and backward passes,
potentially increasing the stability of the algorithm.

Regularization
--------------

A CTIS measures only a handful of projections through the :math:`(x, y, \lambda)`
cube, so the inversion is ill-posed: many different cubes reproduce the
images to within the noise.
MART tends to smear intensity along the projection directions,
and where the smeared projections of unrelated sources intersect they produce
spurious bright features, an artifact which :cite:t:`Parker2022` nicknamed "plaid".
In the reconstructed spectral line profiles, plaid appears as excess intensity
in the wings, since the wavelength direction is the one least constrained by
the few projections.

:cite:t:`Parker2022` suppressed the plaid with an outer loop around MART,
which alternately enhances the contrast of the current guess and convolves it
with the separable smoothing kernel

.. math::

    K_{ijk} = \frac{2^{3 - |i| - |j| - |k|}}{64}, \quad i, j, k = -1, 0, 1,

which is :math:`[1/4, 1/2, 1/4]` along each of the three axes of the cube.

Our implementation takes the slightly different approach of :cite:t:`Silverman1990`,
who alternate every step of an expectation-maximization iteration with a
smoothing step.
After every multiplicative correction, the current guess takes one
gradient-descent step of size :math:`\beta` on the quadratic smoothness penalty

.. math::

    R(\hat{u}) = \frac{1}{2} \sum_k \left( \hat{u}_{k+1} - \hat{u}_k \right)^2,

where :math:`k` indexes the cells along a regularized axis,

.. math::

    \hat{u}_{i+1} = \overline{C}_i \hat{u}_i - \beta \nabla R(\overline{C}_i \hat{u}_i).

That step is a convolution with the kernel :math:`[\beta, 1 - 2 \beta, \beta]`
along the regularized axis.
If more than one axis is regularized, the step is taken along each axis in turn,
which is a convolution with the separable kernel,
so :math:`\beta = 1/4` on all three axes recovers the kernel of :cite:t:`Parker2022` exactly.
The kernel has no negative weights (so positivity is preserved)
as long as :math:`\beta \le 1 / 2`,
and the step conserves the sum of :math:`\hat{u}` along each regularized axis.
When only the wavelength axis is regularized, which is the default,
the radiance integrated over wavelength in every spatial pixel is therefore
unchanged by the regularization, and only the shape of the line profile is smoothed.

The regularization changes the solution the iteration converges to, not just the path to it.
At a fixed point, the multiplicative correction must undo the smoothing step,
:math:`\overline{C} \hat{u} \approx \hat{u} + \beta \nabla R(\hat{u})` to first order in :math:`\beta`,
so the reconstruction accepts a mismatch with the images of order :math:`\beta`
times the relative curvature of the line profile in exchange for smoothness.
The weight is set with :attr:`~ctis.inverters.MartInverter.regularization`
and the regularized axes with :attr:`~ctis.inverters.MartInverter.axis_regularization`.

The contrast-enhancement step of :cite:t:`Parker2022`, which sharpens the guess
to pull the plaid back into the brightest sources, is not implemented.
