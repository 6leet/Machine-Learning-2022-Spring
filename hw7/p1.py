from audioop import avg
from typing import ChainMap
import numpy as np
import cv2
import matplotlib.pyplot as plt

subN = 15
confs = ['centerlight', 'glasses', 'happy', 'leftlight', 'noglasses', 'normal', 'rightlight', 'sad', 'sleepy', 'surprised', 'wink']
faceShape = (231, 195)

def getFaceAndAvg(basePath):
    faces = []
    for i in range(1, subN + 1):
        for conf in confs:
            path = '{}/subject{:0>2d}.{}.pgm'.format(basePath, i, conf)
            face = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if face is not None:
                face = face.reshape(1, -1)[0]
                faces.append(face)

    avgFace = np.sum(faces, axis=0) / len(faces)

    return np.array(faces), avgFace

def showFaces(faces, r, c):
    _, axs = plt.subplots(r, c)
    axs = axs.flatten()
    faces = faces[:len(axs)]
    for face, ax in zip(faces, axs):
        face = face.reshape(faceShape)
        ax.imshow(face, cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()

def getEigen(phi): # diff matrix (all faces - average face)
    Cp = phi @ phi.T
    _ , eigvec = np.linalg.eig(Cp)
    eigFaces = (phi.T @ eigvec).T # .T: same shape as normal faces
    for i in range(len(eigFaces)):
        eigFaces[i] = eigFaces[i] / np.linalg.norm(eigFaces[i])

    showFaces(eigFaces, 5, 5)
    return eigFaces

def transform(faces, eigFaces, avgFace):
    proj = (faces - avgFace) @ eigFaces.T
    recons = proj @ eigFaces + avgFace

    showFaces(recons, 1, 10)
    return proj

faces, avgFace = getFaceAndAvg('Yale_Face_Database/Training')
eigFaces = getEigen(faces - avgFace)
transform(faces, eigFaces[:25], avgFace)
