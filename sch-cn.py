import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import sparse as sparse
from scipy.sparse import linalg as ln
import math as mt

matplotlib.rc('font', size=18)
matplotlib.rc('font', family='Arial')

m=1 #massa
w=1 #frequencia
e=2.718281828459045235360


N = 101 #numero de pontos na malha
sigma = 1j #fazendo h=1 
dt = 0.002 #delta tempo
L = float(10) #tamanho da malha
nsteps = 300 #numero de passos
dx = L/(N-1) #espacamento calculado com base no numero de pontos
nplot = 5 #fazer uma imagem a cada nplots passos
pi=np.pi
l0=np.sqrt(1/(m*w))

#malha

x = np.linspace(0,L,N)
poten = m*w*x**2/2  #m w^2 x^2
r = sigma*dt/dx**2 
h_diag = np.ones(N) / dx**2 + poten
h_non_diag = np.ones(N-1) * (-0.5 /dx**2)
hamiltonian = sparse.diags([h_diag, h_non_diag, h_non_diag], [0, 1, -1])

implicit = (sparse.eye(N) - dt / 2.0j * hamiltonian).tocsc()
explicit = (sparse.eye(N) + dt / 2.0j * hamiltonian).tocsc() 
evolution_matrix = ln.inv(implicit).dot(explicit).tocsr()
 
#consicoes iniciais 
psi = np.asarray([e**(-0.5*(xx-0.5*L)**2/l0**2) for xx in x])


prob = abs(psi)**2
norm = sum(prob)
prob /= norm
psi /= norm**0.5


ymax=[-max(psi)-max(psi)*0.3,max(psi)+max(psi)*0.2]


plt.plot(x,psi,linewidth=2,label="PSI")
plt.plot(x,prob,linewidth=2,label="|PSI|²")
plt.ylim(ymax)
plt.xlabel("x")
plt.legend(loc=1, prop={'size': 10})
plt.title("t = %2.2f"%0)
plt.savefig("./harmonico/harmonico000",format="jpg")
plt.clf()


k = 0
for j in range(nsteps):
    #prina o numero de passos
    print(j)
    
    psi = evolution_matrix.dot(psi)
    prob = abs(psi)**2
    
    norm = sum(prob)
    prob /= norm
    psi /= norm**0.5

    if(j%nplot==0): 
        plt.plot(x,psi,linewidth=2,label="PSI")
        plt.plot(x,prob,linewidth=2,label="|PSI|²")
        plt.ylim(ymax)
        filename = 'harmonico' + str(k+1).zfill(3) + '.jpg';
        plt.xlabel("x")
        plt.legend(loc=1, prop={'size': 10})
        plt.title("t = %2.2f"%(dt*(j+1)))
        plt.savefig("./harmonico/"+filename,format="jpg")
        plt.clf()
        k += 1
