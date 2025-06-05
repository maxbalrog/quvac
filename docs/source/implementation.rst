.. _implementation:

Implementation
==============

Vacuum Emission Amplitude
-------------------------

Here we provide conventions used to implement the calculation of vacuum emission amplitude and signal photon spectrum.

We specify formulas in natural units (:math:`\hbar = c = 1`) but our code uses SI units. We use :math:`g^{\mu\nu}=(-1,1,1,1)`.

.. note::
    We follow the convention/derivation put by [1]_ (Eq. 27 and below) and [2]_ (Section II).

Formalism
^^^^^^^^^

The zero-to-single signal photon transition amplitude to a state with wave vector :math:`k^{\mu}=(\omega,\vec{k})`, with :math:`\omega=|\vec{k}|`,
and transverse polarization vector :math:`\epsilon^{\mu}_{(p)}` is given by

.. math::
    S_{p}(\vec{k}) = \frac{\epsilon^{*\mu}_{p}(k)}{\sqrt{2 k^0}} \int d^4 x e^{-ikx} j_{\mu}(x) |_{k^0=|\vec{k}|}

where

.. math::
    j_{\mu}(x) = 2 \partial^{\nu} \frac{\partial L_{HE}}{\partial F^{\nu\mu}}

is the signal-photon current induced by the macroscopic electromagnetic fields :math:`F^{\mu\nu}`.

The differential number of signal photons is given by

.. math::
    d^3 N_{p}(\vec{k}) = \frac{d^3 k}{(2\pi)^3} |S_{p}(\vec{k})|^2 = \frac{d^3 k}{(2\pi)^3} |\vec\epsilon_{p}(k) \cdot \vec j(k)|.

Wavevector and two orthogonal polarization vectors are defined as

.. math::
    \vec k = \begin{pmatrix}
        \cos\varphi \sin\theta\\ \sin\varphi \sin\theta \\ \cos\theta
    \end{pmatrix},\:
    \vec \epsilon_1 = \begin{pmatrix}
        \cos\varphi \cos\theta\\ \sin\varphi \cos\theta \\ -\sin\theta
    \end{pmatrix},\:
    \vec \epsilon_2 = \begin{pmatrix}
        -\sin\varphi \\ \cos\varphi \\ 0
    \end{pmatrix}

with :math:`\vec k \times \vec\epsilon_1 = \vec\epsilon_2,\: \vec k \times \vec\epsilon_2 = -\vec\epsilon_1`.

Also,

.. math::
    \vec\epsilon_{p}(k) \cdot \vec j(k) = i \sqrt{k^0} \int d^4 x e^{-ikx} [\vec \epsilon_{p}(k) \cdot \vec P - \vec \epsilon_{p+1}(k) \cdot \vec M]

where :math:`\vec P, \vec M` could be written as

.. math::
    \begin{align*}
    \vec P &\simeq -\text{prefactor} [4 \vec E \mathcal F + 7 \vec B \mathcal G],\\
    \vec M &\simeq -\text{prefactor} [4 \vec B \mathcal F - 7 \vec E \mathcal G],\\
    \text{prefactor} &= \sqrt{\frac{\alpha}{\pi}} \frac{m^2}{90 \pi} \left( \frac{e}{m^2}\right)^3
    \end{align*}


Noticing two structures that need to be Fourier transformed, we define

.. math::
    \vec U_1 = \int d^4 x e^{i\omega t - i\vec k \vec x} (4 \vec E \mathcal F + 7 \vec B \mathcal G) \\
    \vec U_2 = \int d^4 x e^{i\omega t - i\vec k \vec x} (4 \vec B \mathcal F - 7 \vec E \mathcal G)

In numerical implementation we do FFT over spatial axes and then integrate over time

.. math::
    \int d^4 x e^{i\omega t - i\vec k \vec x} f(t,x) \rightarrow \sum_{n=0}^{N_t-1} \Delta t e^{i\omega t_n} FFT_3[f(t_n, x)]

Then

.. math::
    \begin{align*}
    S_1 &= -\text{prefactor}\: \cdot i \sqrt{k^0} \: \vec\epsilon_{1}(k) \cdot \vec j(k) = -\text{prefactor}\: \cdot i \sqrt{k^0} [\vec \epsilon_1 \cdot \vec U_1 - \vec \epsilon_2 \cdot \vec U_2], \\
    S_2 &= -\text{prefactor}\: \cdot i \sqrt{k^0} \: \vec\epsilon_{2}(k) \cdot \vec j(k) = -\text{prefactor}\: \cdot i \sqrt{k^0} [\vec \epsilon_2 \cdot \vec U_1 + \vec \epsilon_1 \cdot \vec U_2]
    \end{align*}

In code we define :math:`I_{ij} = \vec \epsilon_i \cdot \vec U_j`.

.. note::
    In SI units the prefactor would be

    .. math::
        \text{prefactor} = \sqrt{\frac{\alpha}{\pi}} \frac{m^2}{90 \pi} \left( \frac{e \hbar}{m^2 c^2}\right)^3 \frac{m^2 c^3}{\hbar^2}

Maxwell propagation
-------------------

Here we specify the procedure for linear Maxwell propagation of electromagnetic fields (it follows Section II of [2]_ ).

We define spatial Fourier transforms as follows

.. math::
    \begin{align*}
    \hat{ \vec E}(t, \vec k) &= \int d^3 x e^{-i \vec k \vec x} \vec E(t, \vec x)\\
    {\vec E}(t, \vec x) &= \int \frac{d^3 k}{(2\pi)^3} e^{i \vec k \vec x} \hat {\vec E}(t, \vec k)
    \end{align*}

and define the vector potential to be spanned by two orthogonal polarization modes

.. math::
    \hat{ \vec A}(t, \vec k) = e^{-i k t} \sum_{p=1}^2 \vec e_p(\vec k) a_p(\vec k).

Having complex model field :math:`\vec E_m (t_0, \vec k)` (either by defining it in the spectral 
domain or Fourier transforming spatial fields), we extract spectral coefficients from them.

.. note::
    In general, extracted spectral amplitudes from model fields are not necessarily orthogonal but 
    we define orthogonal amplitudes by projection.

.. math::
    a_p(\vec k) = e^{i k t_0} \frac{1}{i k} \vec e_p(\vec k) \cdot \hat{\vec E}_m(t_0, \vec k)

From here, fields at other time steps are given by

.. math::
    \begin{align*}
    \hat{\vec E}(t, \vec k) &= e^{-i k t} i k [\vec e_1(\vec k) a_1(\vec k) + \vec e_2(\vec k) a_2(\vec k)], \\
    \hat{\vec B}(t, \vec k) &= e^{-i k t} i k [\vec e_2(\vec k) a_1(\vec k) - \vec e_1(\vec k) a_2(\vec k)].
    \end{align*}

Fields at the spatial domain at timestep :math:`t` are obtained via inverse Fourier transform.

From these complex fields, electromagnetic fields that are used in the vacuum emission amplitude are given by their real part:
:math:`\vec E(t, \vec x) \rightarrow Re\{E(t, \vec x)\}, \vec B(t, \vec x) \rightarrow Re\{B(t, \vec x)\}`.

----

**Notes about numerical implementation**:

1. We slightly modify the formulas for :math:`a_p(\vec k)` and :math:`\hat{\vec E}(t, \vec k)` 
to avoid unnecessary computation (remove factors :math:`i k` and combine two time exponents into one).

2. After the model field projection, its energy might change. We compensate this after the projection.

3. We use DFT to numerically perform continuous Fourier transforms. For grids not starting at 0 there 
is a phase factor mismatch between two transforms.

References
^^^^^^^^^^

.. [1] F. Karbstein. "Probing vacuum polarization effects with high-intensity lasers." 
    Particles 3.1 (2020): 39-61.

.. [2] A. Blinne, et al. "All-optical signatures of quantum vacuum nonlinearities 
    in generic laser fields." PRD 99.1 (2019): 016006.