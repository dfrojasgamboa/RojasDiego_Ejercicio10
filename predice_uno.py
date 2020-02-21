import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

numeros = skdata.load_digits()
target = numeros['target']
imagenes = numeros['images']
n_imagenes = len(target)


data = imagenes.reshape((n_imagenes, -1)) # para volver a tener los datos como imagen basta hacer data.reshape((n_imagenes, 8, 8))

scaler = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.7) # train_size

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

numero = 1
dd = y_train==numero
cov = np.cov(x_train[dd].T)
valores, vectores = np.linalg.eig(cov)
valores = np.real(valores)
vectores = np.real(vectores)
ii = np.argsort(-valores)
valores = valores[ii]
vectores = vectores[:,ii]


dot_mat = np.dot(x_train[y_train==1] , (vectores[:n_pcas]).T)
n_ones, n_pacs = np.shape( dot_mat )
mean_one = [ np.mean(dot_mat[:,i]) for i in range(n_pacs) ]
std_one = [ np.std(dot_mat[:,i]) for i in range(n_pacs) ]

def likelihood(x_test, vec_pcas, mean_one, std_one ):
    dot_mat = np.dot(x_test , vec_pcas.T)
    n_ones, n_pcas = np.shape( dot_mat )
    for i in range(n_pcas):
        chi_squared = (1.0/std_one[i]**2)*sum((dot_mat[:,i]-mean_one[i])**2)
    return np.exp(-chi_squared)