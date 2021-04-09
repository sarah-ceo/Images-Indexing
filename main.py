from vocabulaire import *
from vectorisation import *
from vectorisation_all import *
from tests import *
from classificateurs import *
import time

if __name__ == "__main__":
	#Ce programme va lancer successivement les étapes du TP: 
	#Création de la matrice vocabulaire et du KDTree
	#Vectorisation de toutes les images 
	#(si vous voulez tester la vectorisation sur une seule image pour la partie 3.1, lancez le fichier vectorisation.py)
	#Puis tests sur 8 images pour trouver les images les plus proches
	#Et enfin classificateurs entraînés (5.2)

	matrice = construct_vocabulaire()

	print("\n######################################################################################################")

	tree = create_tree(matrice)
	files, vectors = vectorisation_all(matrice, tree)

	print("\n######################################################################################################")

	tests(matrice, tree, files, vectors)

	print("\n######################################################################################################")

	test_classificateurs(matrice, tree, files, vectors)