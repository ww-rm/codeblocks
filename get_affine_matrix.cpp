// [XY](3, 2) = [UV](3, 3) @ M(3, 2)
// | x1, y1 |   | u1, v1, 1 |   | m11, m12 |
// | x2, y2 | = | u2, v2, 1 | @ | m21, m22 |
// | x3, y3 |   | u3, v3, 1 |   | dx,  dy  |
void GetAffineMatrix(
    double x1, double y1,
    double x2, double y2,
    double x3, double y3,
    double u1, double v1,
    double u2, double v2,
    double u3, double v3,
    Matrix* m
)
{
    double x12 = x1 - x2;
    double y12 = y1 - y2;
    double x23 = x2 - x3;
    double y23 = y2 - y3;
    double u12 = u1 - u2;
    double v12 = v1 - v2;
    double u23 = u2 - u3;
    double v23 = v2 - v3;

    double invdet = 1.0000 / (u12 * v23 - v12 * u23);
    double m11    = invdet * (x12 * v23 - v12 * x23);
    double m12    = invdet * (y12 * v23 - v12 * y23);
    double m21    = invdet * (u12 * x23 - x12 * u23);
    double m22    = invdet * (u12 * y23 - y12 * u23);
    double dx     = x1 - m11 * u1 - m21 * v1;
    double dy     = y1 - m12 * u1 - m22 * v1;

    m->SetElements((REAL)m11, (REAL)m12, (REAL)m21, (REAL)m22, (REAL)dx, (REAL)dy);
}

