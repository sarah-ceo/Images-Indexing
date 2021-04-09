import os
import cv2
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import numpy as np
import matplotlib.pyplot as plt
import time
import random

def sift_search():
	print("\nRECHERCHE DES SIFT DANS LA BASE D'ENTRAINEMENT (~15s)")
	start = time.time()

	#On crée l'objet SIFT et on initialise la matrice de descripteurs SIFT
	sift = cv2.SIFT_create(200)
	sift_desc = np.empty((0, 128))
	kps = []

	#Voici les 4 classes choisies pour l'entraînement, soit un total de 235 images dans les dossiers train
	folders = ["./camera/train/", "./umbrella/train/", "./butterfly/train/", "./snoopy/train/"]

	i = 0
	for folder in folders:
		for f in os.listdir(folder):
			filepath = os.path.join(folder, f)
			if f.endswith(".jpg"):
				#On fait des print intermédiaires pour s'assurer que tout va bien
				i+=1
				if i%100==0:
					print(i, " / 235 images traitées")

				#On charge l'image et on la convertit en niveaux de gris
				img = cv2.imread(filepath)
				gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

				#On détecte les points SIFT
				keypoints, descriptors = sift.detectAndCompute(gray, None)

				#On affiche les SIFT trouvés sur la première image à titre d'exemple
				if i == 1:
					img_drawn = cv2.drawKeypoints(gray, keypoints, img, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
					plt.title("Exemples de SIFT trouvés")
					plt.axis("off")
					plt.imshow(img_drawn)
					plt.show()

				#On garde aussi les portions d'images correspondantes aux keypoints
				for keypoint in keypoints:
					x = int(keypoint.pt[0])
					y = int(keypoint.pt[1])
					size = int(keypoint.size+10)
					portion = gray[x-size:x+size, y-size:y+size]
					kps.append(portion)

				#On ajoute les descripteurs trouvés à notre matrice de descripteurs SIFT
				sift_desc = np.vstack((sift_desc, descriptors))

	#Pour info
	elapsed = time.time() - start
	print("Temps de traitement : ", time.strftime("%H:%M:%S", time.gmtime(elapsed)))
	print("Shape des descripteurs: ", sift_desc.shape)
	return sift_desc, kps

def n_search(sift_desc):
	print("\nRECHERCHE DE N (De 1 à 100)")

	#On va chercher la variance intra-classe selon un N variant de 1 à la moitié de notre nombre de SIFT
	variances = []
	N = range(1, 100)
	start = time.time()

	for n in N:
		print("Test avec N =", n)
		variance = 0
		kmeans = KMeans(n).fit(sift_desc)
		variances.append(kmeans.inertia_)

	elapsed = time.time() - start
	print("Temps de traitement : ", time.strftime("%H:%M:%S", time.gmtime(elapsed)))

	#On plot le tout pour appliquer la elbow method
	plt.figure()
	plt.plot(N, variances)
	plt.xlabel("Nombres de clusters")
	plt.ylabel("Variance intra-classe")
	plt.title("Elbow method")
	plt.savefig("./out/elbow_method_results.png")
	plt.show()
	print("Image sauvegardée (./out/elbow_method_results.png)")

def save_matrix(N, sift_desc, kps):
	print("\nCREATION DE LA MATRICE DE VOCABULAIRE AVEC LE BEST N =", N)
	start = time.time()
	#On effectue la clusterisation avec KMeans et on sauvegarde les centres de clusters
	kmeans = KMeans(N).fit(sift_desc)
	elapsed = time.time() - start
	print("Temps de traitement : ", time.strftime("%H:%M:%S", time.gmtime(elapsed)))
	print("Variance intra-classe : ", kmeans.inertia_)
	matrice = kmeans.cluster_centers_
	np.savetxt("./out/matrice_vocabulaire.out", matrice)
	print("Matrice de vocabulaire sauvegardée (out/matrice_vocabulaire.out)")

	#On sauvegarde deux portions d'images pour chaque cluster
	for i in range(N):
		idx = np.where(kmeans.labels_ == i)
		kp1 = kps[idx[0][0]]
		kp2 = kps[idx[0][random.randint(1,len(idx[0])-1)]]
		if kp1.size:
			cv2.imwrite("./out/clusters_snippets/cluster"+str(i)+"_portion1.png", kp1)
		if kp2.size:
			cv2.imwrite("./out/clusters_snippets/cluster"+str(i)+"_portion2.png", kp2)
	print("Portions d'images sauvegardées dans le dossier out/clusters_snippets/")
	return matrice

def construct_vocabulaire():
	descriptors, keypoints = sift_search()

	answer = input("\nSouhaitez-vous effectuer la recherche du N optimal pour le KMeans (~2h sur nos machines)?  y/n \n")
	if answer == "y":
		n_search(descriptors)

	#Le meilleur nombre de clusters de points SIFT pour le KMeans d'après la "Elbow Method"
	best_N = 24

	answer = input("\nSouhaitez-vous recréer la matrice_vocabulaire? (~40s sur nos machines) y/n \nCela risque de donner des résultats différents de ceux énoncés dans notre fichier réponse... \n")
	if answer == "y":
		matrice = save_matrix(best_N, descriptors, keypoints)
	else:
		matrice = np.loadtxt("./out/matrice_vocabulaire.out")
	return matrice

if __name__ == "__main__" :
	construct_vocabulaire()
