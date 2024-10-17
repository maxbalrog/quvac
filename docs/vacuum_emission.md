# Vacuum Emission Amplitude

Here we provide conventions used to implement the calculation of vacuum emission amplitude and signal photon spectrum.

We specify formulas in natural units ($\hbar = c = 1$) but our code uses SI units. We use $g^{\mu\nu}=(-1,1,1,1)$.

>We follow the convention/derivation put by [[1]](#1) (Eq. 27 and below) and [[2]](#2) (Section II).

## Formalism
The zero-to-single signal photon transition amplitude to a state with wave vector $k^{\mu}=(\omega,\vec{k})$, with $\omega=|\vec{k}|$, and transverse polarization vector $\epsilon^{\mu}_{(p)}$ is given by
$$S_{p}(\vec{k}) = \frac{\epsilon^{*\mu}_{p}(k)}{\sqrt{2 k^0}} \int d^4 x e^{-ikx} j_{\mu}(x) |_{k^0=|\vec{k}|}$$
where
$$j_{\mu}(x) = 2 \partial^{\nu} \frac{\partial L_{HE}}{\partial F^{\nu\mu}}$$
is the signal-photon current induced by the macroscopic electromagnetic fields $F^{\mu\nu}$.

The differential number of signal photons is given by
$$
d^3 N_{p}(\vec{k}) = \frac{d^3 k}{(2\pi)^3} |S_{p}(\vec{k})|^2 = \frac{d^3 k}{(2\pi)^3} |\vec\epsilon_{p}(k) \cdot \vec j(k)|.
$$

Wavevector and two orthogonal polarization vectors are defined as
$$
\vec k = \begin{pmatrix}
    \cos\varphi \sin\theta\\ \sin\varphi \sin\theta \\ \cos\theta
\end{pmatrix},\:
\vec \epsilon_1 = \begin{pmatrix}
    \cos\varphi \cos\theta\\ \sin\varphi \cos\theta \\ -\sin\theta
\end{pmatrix},\:
\vec \epsilon_2 = \begin{pmatrix}
    -\sin\varphi \\ \cos\varphi \\ 0
\end{pmatrix}
$$
with $\vec k \times \vec\epsilon_1 = \vec\epsilon_2,\: \vec k \times \vec\epsilon_2 = -\vec\epsilon_1$.

Also,
$$
\vec\epsilon_{p}(k) \cdot \vec j(k) = i \sqrt{k^0} \int d^4 x e^{-ikx} [\vec \epsilon_{p}(k) \cdot \vec P - \vec \epsilon_{p+1}(k) \cdot \vec M]
$$
where $\vec P, \vec M$ could be written as
$$
\begin{align*}
\vec P &\simeq -\text{prefactor} [4 \vec E \mathcal F + 7 \vec B \mathcal G],\\
\vec M &\simeq -\text{prefactor} [4 \vec B \mathcal F - 7 \vec E \mathcal G],\\
\text{prefactor} &= \sqrt{\frac{\alpha}{\pi}} \frac{m^2}{90 \pi} \left( \frac{e}{m^2}\right)^3
\end{align*}
$$

Noticing two structures that need to be Fourier transformed, we define
$$
\vec U_1 = \int d^4 x e^{i\omega t - i\vec k \vec x} (4 \vec E \mathcal F + 7 \vec B \mathcal G) \\
\vec U_2 = \int d^4 x e^{i\omega t - i\vec k \vec x} (4 \vec B \mathcal F - 7 \vec E \mathcal G)
$$
In numerical implementation we do FFT over spatial axes and then integrate over time
$$
\int d^4 x e^{i\omega t - i\vec k \vec x} f(t,x) \rightarrow \sum_{n=0}^{N_t-1} \Delta t e^{i\omega t_n} FFT_3[f(t_n, x)]
$$

Then
$$
\begin{align*}
S_1 &= -\text{prefactor}\: \cdot i \sqrt{k^0} \: \vec\epsilon_{1}(k) \cdot \vec j(k) = -\text{prefactor}\: \cdot i \sqrt{k^0} [\vec \epsilon_1 \cdot \vec U_1 - \vec \epsilon_2 \cdot \vec U_2], \\
S_2 &= -\text{prefactor}\: \cdot i \sqrt{k^0} \: \vec\epsilon_{2}(k) \cdot \vec j(k) = -\text{prefactor}\: \cdot i \sqrt{k^0} [\vec \epsilon_2 \cdot \vec U_1 + \vec \epsilon_1 \cdot \vec U_2]
\end{align*}
$$

In code we define $I_{ij} = \vec \epsilon_i \cdot \vec U_j$.

> In SI units the prefactor would be
> $$
> \text{prefactor} = \sqrt{\frac{\alpha}{\pi}} \frac{m^2}{90 \pi} \left( \frac{e \hbar}{m^2 c^2}\right)^3 \frac{m^2 c^3}{\hbar^2}
> $$


## References
<a id="1">[1]</a>
F. Karbstein. "Probing vacuum polarization effects with high-intensity lasers." Particles 3.1 (2020): 39-61.

<a id="2">[2]</a>
A. Blinne, et al. "All-optical signatures of quantum vacuum nonlinearities in generic laser fields." PRD 99.1 (2019): 016006.