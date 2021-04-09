import sys
import cv2
import numpy as np
from sklearn.neighbors import KDTree
import pickle

def vectorisation(image, matrice, tree):
	sift = cv2.SIFT_create(200)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#On détecte les points SIFT de l'image
	keypoints, descriptors = sift.detectAndCompute(gray, None)

	#On crée le vecteur qui comptabilisera le nombre de fois où un mot de vocabulaire visuel
	# aura été trouvé comme plus proche voisin d'un descripteur
	vector = np.zeros((matrice.shape[0],1))

	#On effectue la recherche des plus proches voisins avec KDTree
	_, ind = tree.query(descriptors)

	#On remplit le vecteur et on le retourne
	for i in range(matrice.shape[0]):
		vector[i] = np.count_nonzero(ind==i)
	return vector.astype(int)

def create_tree(matrice):
	tree = KDTree(matrice)
	with open("./out/tree.p", "wb") as f:
		pickle.dump(tree, f)
	print("KDTree sauvegardé (out/tree.p)")
	return tree


if __name__ == "__main__":
	#Si l'utilisateur ne donne pas d'argument en entrée du fichier, on lit l'image 001 de la premiere classe
	if len(sys.argv) < 2:
		filepath = "./camera/test/image_0001.jpg"
	else:
		filepath = sys.argv[1]
	image = cv2.imread(filepath)

	#On récupère la matrice de vocabulaire
	matrice = np.loadtxt("./out/matrice_vocabulaire.out")
	print("Fichier matrice_vocabulaire.out chargé", matrice.shape)

	tree = create_tree(matrice)

	print("VECTORISATION DE L'IMAGE : ", filepath)
	v = vectorisation(image, matrice, tree)
	print("Vecteur : ")
	print(v)