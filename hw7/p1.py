from audioop import avg
import numpy as np
import cv2
import matplotlib.pyplot as plt

subN = 15
confs = ['centerlight', 'glasses', 'happy', 'leftlight', 'noglasses', 'normal', 'rightlight', 'sad', 'sleepy', 'surprised', 'wink']
faceShape = (100, 100)

def getFace(basePath):
    faces = []
    labels = []
    for label in range(1, subN + 1):
        for conf in confs:
            path = '{}/subject{:0>2d}.{}.pgm'.format(basePath, label, conf)
            face = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if face is not None:
                face = cv2.resize(face, faceShape)
                face = face.reshape(1, -1)[0]
                faces.append(face)
                labels.append(label)

    return np.array(faces).T, np.array(labels)

def showFaces(faces, r, c):
    _, axs = plt.subplots(r, c)
    axs = axs.flatten()
    faces = faces.T
    faces = faces[:len(axs)]
    for face, ax in zip(faces, axs):
        face = face.reshape(faceShape)
        ax.imshow(face, cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

def knn(trainData, testData, k):
    projs, labels = trainData
    testProjs, testLabels = testData

    dists = []
    for i in range(len(testLabels)):
        dist = []
        for j in range(len(labels)):
            d = np.linalg.norm(testProjs[:, i] - projs[:, j])
            dist.append((d, labels[j]))
        dists.append(dist)
    
    acc = 0
    for dist, gtLabel in zip(dists, testLabels):
        sortByDist = sorted(dist, key=lambda tup: tup[0])
        knearest = [label for _, label in sortByDist[:k]]
        pdLabel = np.bincount(knearest).argmax()
        acc += (pdLabel == gtLabel)
    acc /= len(testLabels)
    
    print('Accuracy: {}'.format(acc))

def pca(X, todim=None):
    # X_mean = np.mean(X, axis=1).reshape(-1, 1)
    meanx = np.mean(X, axis=1).reshape(-1, 1)
    centerX = X - meanx

    eigval, eigvec = np.linalg.eig(centerX.T @ centerX)
    sortIdx = np.argsort(-eigval)
    sortIdx = sortIdx[:todim] if todim is not None else sortIdx[sortIdx > 0]

    eigval=eigval[sortIdx]
    # # from X.T@X eigenvector to X@X.T eigenvector
    eigvec = centerX @ eigvec[:, sortIdx]
    eigvec = eigvec / np.linalg.norm(eigvec,axis=0)

    return eigval, eigvec, meanx

def getSbAndSw(X, y):
    d, _ = X.shape
    yCnts = np.bincount(y)
    Sb = np.zeros((d, d))
    Sw = np.zeros((d, d))
    head = 0
    meanx = np.mean(X, axis=1).reshape(-1, 1)
    for ycnt in yCnts[1:]:
        classI = np.array(X[:, head:head + ycnt])
        avgI = np.mean(classI, axis=1).reshape(-1, 1)
        Sb += ycnt * (avgI - meanx) @ (avgI - meanx).T
        Sw += (classI - avgI) @ (classI - avgI).T
        head += ycnt

    return Sb, Sw

def lda(X, y, todim=None):
    Sb, Sw = getSbAndSw(X, y)

    eigVal, eigVec = np.linalg.eig(np.linalg.inv(Sw) @ Sb)
    sortIdx=np.argsort(-eigVal.real)
    sortIdx = sortIdx[:todim] if todim is not None else sortIdx[:-1]

    eigVal = np.asarray(eigVal[sortIdx].real, dtype='float')
    eigVec = np.asarray(eigVec[:, sortIdx].real, dtype='float')

    return eigVal, eigVec

def eigenface(faces, labels):
    _, eigFaces, avgFace = pca(faces, 25)
    showFaces(eigFaces, 5, 5)

    projs = eigFaces.T @ (faces - avgFace)
    recovs = eigFaces @ projs + avgFace
    np.random.shuffle(recovs.T)
    showFaces(recovs, 1, 10)
    
    testFaces, testLabels = getFace(testPath)
    testProjs = eigFaces.T @ (testFaces - avgFace)
    knn((projs, labels), (testProjs, testLabels), subN)

def fisherface(faces, labels):
    _, eigFacesPca, avgFace = pca(faces, 31)
    pcaProjs = eigFacesPca.T @ (faces - avgFace)

    _, eigVecLda = lda(pcaProjs, labels, 25)
    print(eigFacesPca.shape, eigVecLda.shape)
    eigFaces = eigFacesPca @ eigVecLda
    showFaces(eigFaces, 5, 5)

    projs = eigFaces.T @ faces
    recovs = eigFaces @ projs + avgFace
    np.random.shuffle(recovs.T)
    showFaces(recovs, 1, 10)

    testFaces, testLabels = getFace(testPath)
    testProjs = eigFaces.T @ testFaces
    knn((projs, labels), (testProjs, testLabels), 11)

def kernelPCA(faces, labels):
    


trainPath = 'Yale_Face_Database/Training'
testPath = 'Yale_Face_Database/Testing'

faces, labels = getFace(trainPath)
