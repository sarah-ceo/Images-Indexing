from vectorisation import vectorisation
import numpy as np
import os
import pickle
from sklearn.neighbors import KDTree
import cv2
import random
import matplotlib.pyplot as plt

def tests(matrice, tree, files, vectors):
	#On met en forme nos vecteurs
	X = np.squeeze(np.array(vectors))

	#On va utiliser un autre KDTree pour trouver nos deux images les plus proches
	tree2 = KDTree(X)

	#On teste sur une image de chaque classe, dans train et dans test
	train_images = ["./butterfly/train/image_0010.jpg","./camera/train/image_0010.jpg","./snoopy/train/image_0010.jpg","./umbrella/train/image_0010.jpg"]
	test_images = ["./butterfly/test/image_0001.jpg","./camera/test/image_0001.jpg","./snoopy/test/image_0001.jpg","./umbrella/test/image_0001.jpg"]

	#On teste sur les images d'entraînement et on affiche le tout avec matplotlib
	fig_train = plt.figure()
	plt.title("Images de train (1ère colonne) et leurs 2 plus proches voisines")
	plt.axis("off")

	print("\nTESTS SUR LES IMAGES D'ENTRAINEMENT ")
	for i,f in enumerate(train_images):
		chosen_image = cv2.imread(f)
		fig_train.add_subplot(4,3,i*3+1)
		plt.axis("off")
		plt.imshow(chosen_image)

		v = vectorisation(chosen_image, matrice, tree)
		Y = np.transpose(np.array(v))
		_, ind = tree2.query(Y, 3)
		print("\nPour ", f, ", les deux images les plus proches sont : ", files[ind[0][1]], "et", files[ind[0][2]])

		fig_train.add_subplot(4,3,i*3+2)
		plt.axis("off")
		plt.imshow(cv2.imread(files[ind[0][1]]))
		fig_train.add_subplot(4,3,i*3+3)
		plt.axis("off")
		plt.imshow(cv2.imread(files[ind[0][2]]))

	plt.savefig("./out/test_train.png")
	plt.show()
	plt.close(fig_train)
	print("Image sauvegardée (./out/test_train.png)")

	#On teste sur les images de test et on affiche le tout avec matplotlib

	fig_test = plt.figure()
	plt.title("Images de test (1ère colonne) et leurs 2 plus proches voisines")
	plt.axis("off")

	print("\nTESTS SUR LES IMAGES DE TEST")
	for i,f in enumerate(test_images):
		chosen_image = cv2.imread(f)
		fig_test.add_subplot(4,3,i*3+1)
		plt.axis("off")
		plt.imshow(chosen_image)

		v = vectorisation(chosen_image, matrice, tree)
		Y = np.transpose(np.array(v))
		_, ind = tree2.query(Y, 2)
		print("\nPour ", f, ", les deux images les plus proches sont : ", files[ind[0][0]], "et", files[ind[0][1]])

		fig_test.add_subplot(4,3,i*3+2)
		plt.axis("off")
		plt.imshow(cv2.imread(files[ind[0][0]]))
		fig_test.add_subplot(4,3,i*3+3)
		plt.axis("off")
		plt.imshow(cv2.imread(files[ind[0][1]]))

	plt.savefig("./out/test_test.png")
	plt.show()
	plt.close(fig_test)
	print("Image sauvegardée (./out/test_test.png)")



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

	tests(matrice, tree, files, vectors)

