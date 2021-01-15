import numpy as np
'''
   m1 m2 m3 m4
u1 4  3  -  -
u2 5  4  -  4
u3 2  1  3  3
u4 -  -  3  -
colloborative filtering->some k similar users and it will compute 0.8
3.2+b
baseline-b=U+bi+bx, bi=av(m2)-U, bx=av(u2)-U
svd->singular value decomposition
A=n*m
u=n*r,sig=r*r,V=r*m
action,drama,adventure r=3
U  action drama adventure
u1  0.6     0.3   0
u2
u3
u4
sig
12 0 0
0  5 0
0  0  1
V       m1  m2 m3 m4
action  56
adv      32
dr       12
A=C*U*R
C-column k
U-pseudo inverse matrix-k*K
R-row k
retaining->90% -> sum=1+25+144, suma=25+144, suma/sum*100
'''
def pearson_sim(M, x, y):
    """
    Pearson correlation coefficient of two rows M(x) and M(y)

    Input:
    M (numpy.ndarray): Input Matrix
    x (int): Index of first item
    y (int): Index of second item
    """
    x_mean = sum(M[x])/np.count_nonzero(M[x])
    y_mean = sum(M[y])/np.count_nonzero(M[y])

    numerator = 0
    denom_x = 0
    denom_y = 0

    for i in range(len(M[x])):
        if M[x][i] != 0 and M[y][i] != 0:
            numerator += (M[x][i] - x_mean)*(M[y][i] - y_mean)
            denom_x += (M[x][i] - x_mean)**2
            denom_y += (M[y][i] - y_mean)**2
        elif M[x][i] == 0 and M[y][i] != 0:
            denom_y += (M[y][i] - y_mean)**2
        elif M[x][i] != 0 and M[y][i] == 0:
            denom_x += (M[x][i] - x_mean)**2

    return numerator/np.sqrt(denom_x * denom_y)
