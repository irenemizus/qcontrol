__kernel void complex_mult(cdouble_t *x, cdouble_t *y)
{
    x[i] = cdouble_mul(x[i], y[i])
}

__kernel void complex_double_mult_add(double *v, cdouble_t *psi, cdouble_t *phi)
{
    phi[i] = cdouble_add(cdouble_mul(psi[i], cdouble_new(v[i], 0.0)), phi[i])
}

__kernel void prop_recurr_part_cl(cdouble_t *psi, cdouble_t *phi, cdouble_t dvj)
{
    psi[i] = cdouble_add(cdouble_mul(phi[i], dvj), psi[i])
}

__kernel void hamil2D_half_cl(cdouble_t *phi_d, cdouble_t *psi_d, cdouble_t *psi_nd, double eL, double E)
{
    phi_d[i] = cdouble_sub(
                    cdouble_sub(
                        phi_d[i], cdouble_mul(
                                        psi_d[i], cdouble_new(eL, 0.0)
                                        )
                        ), cdouble_mul(
                            psi_nd[i], cdouble_new(E, 0.0)
                            )
                    )
}

__kernel void residum_half_cl(cdouble_t *phi_d, cdouble_t *psi_d, double coef1, double coef2, double xp)
{
    phi_d[i] = cdouble_sub(
                    cdouble_sub(
                        cdouble_mul(
                            phi_d[i], cdouble_new(coef1, 0.0)
                            ), cdouble_mul(
                                    psi_d[i], cdouble_new(coef2, 0.0)
                                    )
                        ), cdouble_mul(
                            psi_d[i], cdouble_new(xp, 0.0)
                            )
                    )
}