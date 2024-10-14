import math
n_dim=[32]
M = 256
N=4096
k_dim = [32]
m_dim = [16]
K=1024
a=1e9
result = []
for n in n_dim:
    for m in m_dim:
        for k in k_dim:
            print(2*m*k*n/170>(m*k+k*n)/1)
            # print(m*k+k*n)
            t = min(2*m*k*n/170,(m*k+k*n)/1) + math.floor(K/k) * max(2*m*k*n/170,(m*k+k*n)/1)
            #print(t*math.floor(M/m))
            if t*math.floor(M/m)*math.floor(N/n)<a:
                a=t*math.floor(M/m)*math.floor(N/n)
                result = [m,n,k]
                print(a)
print(result)