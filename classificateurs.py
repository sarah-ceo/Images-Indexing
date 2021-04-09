from vectorisation import vectorisation
from sklearn.svm import NuSVC
import pickle
import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt


def create_classificateurs(files, vectors, labels):
	#On crée les trois classificateurs
	clfA = NuSVC()
	clfB = NuSVC()
	clfC = NuSVC()

	#On crée nos données pour l'entrainement de ces classificateurs
	XA = []
	YA = []
	XB = []
	YB = []
	XC = []
	YC = []

	for i in range(len(files)):
		f = files[i]
		v = vectors[i]
		XA.append(v)
		if labels[0] in f or labels[1] in f:
			YA.append(0)
			XB.append(v)
			if labels[0] in f:
				YB.append(0)
			else:
				YB.append(1)
		else:
			YA.append(1)
			XC.append(v)
			if labels[2] in f:
				YC.append(0)
			else:
				YC.append(1)

	#On entraîne
	clfA.fit(XA, YA)
	clfB.fit(XB, YB)
	clfC.fit(XC, YC)

	return clfA, clfB, clfC

def classify(clfA, clfB, clfC, X, labels):
	answerA = clfA.predict(X)[0]
	print("A = ", answerA, end=", ")

	if answerA == 0:
		answerB = clfB.predict(X)[0]
		print("B = ", answerB, end=", ")
		print("Classe : ", labels[answerB])
		classe = labels[answerB]

	else:
		answerC = clfC.predict(X)[0]
		print("C = ", answerC, end=", ")
		print("Classe :", labels[2+answerC])
		classe = labels[2+answerC]

	return classe


def test_classificateurs(matrice, tree, files, vectors):
	print("\nTESTS AVEC LES CLASSIFIEURS NuSVC")
	print("A: 0->butterfly/camera, 1->snoopy/umbrella")
	print("B: 0->butterfly, 1->camera")
	print("C: 0->snoopy, 1->umbrella")
	labels = ["butterfly", "camera", "snoopy", "umbrella"]
	clfA, clfB, clfC = create_classificateurs(files, vectors, labels)

	#On teste sur les mêmes images que dans la partie 4
	train_images = ["./butterfly/train/image_0010.jpg","./camera/train/image_0010.jpg","./snoopy/train/image_0010.jpg","./umbrella/train/image_0010.jpg"]
	test_images = ["./butterfly/test/image_0001.jpg","./camera/test/image_0001.jpg","./snoopy/test/image_0001.jpg","./umbrella/test/image_0001.jpg"]


	#On teste sur les images d'entraînement et on affiche le tout avec matplotlib
	fig_train = plt.figure()
	plt.axis("off")

	print("\nSUR LES IMAGES D'ENTRAINEMENT")
	for i,f in enumerate(train_images):
		chosen_image = cv2.imread(f)
		print("\nPour ", f, end=", ")
		X = np.transpose(vectorisation(chosen_image, matrice, tree))
		classe = classify(clfA, clfB, clfC, X, labels)

		ax_train = fig_train.add_subplot(4, 1, i+1)
		ax_train.title.set_text(classe)
		ax_train.axis("off")
		ax_train.imshow(chosen_image)

	plt.savefig("./out/classificateur_train.png")
	plt.show()
	plt.close(fig_train)
	print("Image sauvegardée (./out/classificateur_train.png)")

	#On teste sur les images de test et on affiche le tout avec matplotlib
	fig_test = plt.figure()
	plt.axis("off")

	print("\nSUR LES IMAGES DE TEST")
	for i,f in enumerate(test_images):
		chosen_image = cv2.imread(f)
		print("\nPour ", f, end=", ")
		X = np.transpose(vectorisation(chosen_image, matrice, tree))
		classe = classify(clfA, clfB, clfC, X, labels)

		ax_test = fig_test.add_subplot(4, 1, i+1)
		ax_test.title.set_text(classe)
		ax_test.axis("off")
		ax_test.imshow(chosen_image)

	plt.savefig("./out/classificateur_test.png")
	plt.show()
	plt.close(fig_test)
	print("Image sauvegardée (./out/classificateur_test.png)")

if __name__ == "__main__":

	#On récupère la matrice de vocabulaire
	matrice = np.loadtxt("./out/matrice_vocabulaire.out")
	print("Fichier matrice_vocabulaire.out chargé")

	#On récupère le KDTree 
	with open("./out/tree.p", "rb") as f:
		tree = pickle.load(f)
	print("Fichier tree.p chargé")

	#On récupère les fichiers de vectorisation
	with open("./out/vectorisation_train_files.p", "rb") as f:
		files = pickle.load(f)
	with open("./out/vectorisation_train_vectors.p", "rb") as f:
		vectors = pickle.load(f)
	print("Fichiers vectorisation_train_files.p & vectorisation_train_vectors.p chargés")

	test_classificateurs(matrice, tree, files, vectors)

