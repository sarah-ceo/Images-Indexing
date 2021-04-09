from vectorisation import vectorisation
import numpy as np
import os
import pickle
from sklearn.neighbors import KDTree
import cv2
import time

def vectorisation_all(matrice, tree):
	#On crée les listes dans lesquelles on va stocker les résultats
	files = []
	vectors = []
	folders = [ "./butterfly/train/", "./camera/train/", "./snoopy/train/", "./umbrella/train/"]

	#On applique la recherche sur chaque image de train et on stocke les résultats dans le dictionnaire
	print("\nVECTORISATION DES IMAGES D'ENTRAINEMENT (~10s)")
	start = time.time()
	i = 0
	for folder in folders:
		for f in os.listdir(folder):
			filepath = os.path.join(folder, f)
			if f.endswith(".jpg"):
				#On fait des print intermédiaires pour s'assurer que tout va bien
				i+=1
				if i%100==0:
					print(i, " / 235 images traitées")

				image = cv2.imread(filepath)
				v = vectorisation(image, matrice, tree)
				files.append(filepath)
				vectors.append(v)

	vectors = np.squeeze(np.array(vectors))
	#On sauvegarde les deux listes (noms de fichiers et vecteur correspondant)
	with open("./out/vectorisation_train_files.p", "wb") as f:
		pickle.dump(files, f)
	with open("./out/vectorisation_train_vectors.p", "wb") as f:
		pickle.dump(vectors, f)

	elapsed = time.time() - start
	print("Temps de traitement : ", time.strftime("%H:%M:%S", time.gmtime(elapsed)))
	print("Vecteurs et noms des fichiers sauvegardés (out/vectorisation_train_vectors.p & out/vectorisation_train_files.p)")
	return files, vectors


if __name__ == "__main__":

	#On récupère la matrice de vocabulaire
	matrice = np.loadtxt("./out/matrice_vocabulaire.out")
	print("Fichier matrice_vocabulaire.out chargé")

	#On récupère le KDTree
	with open("./out/tree.p", "rb") as f:
		tree = pickle.load(f)
	print("Fichier tree.p chargé")

	vectorisation_all(matrice, tree)

	

	