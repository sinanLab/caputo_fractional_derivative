# caputo_fractional_derivative
A python function Caputo Fractional Derivative or Caputo Fractional Differential Operator with Pytorch\
It was introduced by Michele Caputo in his 1967 paper. [1] In contrast to the Riemann–Liouville fractional derivative, when solving differential equations using Caputo's definition, it is not necessary to define the fractional order initial conditions. Caputo's definition is illustrated as follows, where again $n = \lceil \alpha \rceil$:
$${}^C D_t^\alpha = \frac{1}{\Gamma(n-\alpha)}\int^t_0 \frac{f^{n}(\tau)}{(t-\tau)^{\alpha+1-n}}d\tau$$\

[1] Caputo, Michele. "Linear models of dissipation whose Q is almost frequency independent—II." Geophysical Journal International 13.5 (1967): 529-539.
