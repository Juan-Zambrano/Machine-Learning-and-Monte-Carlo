import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

def clean_data():
    with open('PCA_MATRIX.txt') as f:
        df = [[d.replace('[','').replace('   ',',').replace(' ',',').replace(',,',',').replace('+0.j','').split(',')[-3:] for d in l.split(']') ]for l in f.readlines()][0]
        xyz_cleaned = [[float(c) for c in r ] for r in df[:-1]]
    return xyz_cleaned

def t_p_cleaned(xyz_cleaned):
    return [[math.atan(y/x),math.atan(z/math.sqrt(x**2+y**2)) ] for (x,y,z) in xyz_cleaned]

def plotmc(theta,phi):
    T = np.linspace(-math.pi/2,math.pi/2,50)
    P = np.linspace(-math.pi/2,math.pi/2,50)
    H = np.array([[math.cos(t-theta)*math.cos(p-phi) for p in P] for t in T])
    print(H)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.contour3D(T, P, H, 50, cmap='binary')
    plt.show()

def main():
    clean_data()
    data = t_p_cleaned(clean_data())
    plotmc(*data[2]) #0
    plotmc(*data[17]) #45
    plotmc(*data[30]) #90
    plotmc(*data[52]) #135
    plotmc(*data[75]) #180
    plotmc(*data[91]) #225
    plotmc(*data[102]) #270
    plotmc(*data[130]) #315
    

main()


