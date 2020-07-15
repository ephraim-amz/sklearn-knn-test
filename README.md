# sklearn-knn-test
## Résumé
 Ce repository a pour objectif de montrer l'utilisation d'un algorithme de reconnaissance faciale : Le _KNN_ (algorithme du plus proche voisin)

Les images utilisés pour l'éxecution de cet algorithme seront des photos de George Bush, ex président des États-Unis ainsi que Colin Powell, ex secrétaire d'état.
[Définition KN](https://fr.wikipedia.org/wiki/M%C3%A9thode_des_k_plus_proches_voisins "Méthode des k plus proches voisins")
___
## Étapes du code

## 1. Importation des librairies utiles
```python
import pandas as pd
import numpy as np
import numpy.linalg as linalg
import matplotlib.pylab as pylab
```
## 2. Importation des images utilisés à partir de la librairie sklearn
```python
from sklearn.datasets import fetch_lfw_people
lfw_dataset = fetch_lfw_people(min_faces_per_person=200) # To keep people who got 200 images or more
import time as time # To calculate time execution
from sklearn.model_selection import train_test_split # To split data
```
___
## 3. Affichage des résultats
```python
# We only show the first images of the X dataset, with the corresponding names
pylab.figure(figsize=(20,20))
for i in range(15):
    pylab.subplot(5,5,i+1)
    pylab.imshow(X[i,:].reshape((img_height, img_width)), cmap=pylab.plt.cm.gray) # figure's display
    pylab.title(names[y[i]])
pylab.show()
```
![Bush and Powell](bush_powell.png)
The labels of above images are : [1 1 1 1 1 1 1 1 1 0 0 0 1 0 1] (y=1) for George W Bush and (y=0) for Colin Powell
___
## 4. Affichage des fonctions utilisés
The `distance` function take 2 images (as vectors) and return the euclidian distance between these 2 images.
```python
def distance(I1,I2):
    return sqrt(sum((I1-I2)**2))
```

```python
def normalise(M):
    n,p=M.shape
    N=zeros((n,p))
    for i in range(p):
        N[:,i]=(M[:,i]-mean(M[:,i]))/std(M[:,i]) # We normalize each row
    return N
```

```python
def correlation(M):
    Z= normalise(M)
    n=Z.shape[0]
    return 1/n*dot(Z.T,Z)
```

```python
def acp(M):
    n,p=M.shape
    R= correlation(M)
    valtemp, vectemp = eigh(R) # eigenvalues and eigenvectors of the correlation matrix
    val = sort(valtemp)[::-1] # To sort eigenvalues by decreasing order
    index = argsort(valtemp)[::-1] # Rearrangement index of eigenvalues by decreasing order
    P=zeros((p,p)) # matrice de changement de base ordonnée en fonction des valeurs propres
    for i in range(p):
        P[:,i]=vectemp[:,index[i]]
    C=dot(normalise(M),P)
    return val, P, C
```
```python
def dissimilarite(I1,I2):
    return 1-corrcoef(I1,I2)[0,1]**2
```

___ 