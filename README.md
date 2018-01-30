# README: A non parametric approach to modeling Overlapping Clusters

Ce fichier contient:
* Un notebook IOMM_algo_synthetic_data_K_finite_final.ipynb pour compiler le Mixture Model avec des données synthétiques
* Un notebook OMM_movies_vf.ipynb pour compiler le Mixture Model avec la base de données des films
* Une classe omm.py contenant les fonctions relatives à l’algorithme Mixture Model avec K fini
* Un notebook IOMM_algo_synthetic_data_K_infinite_final.ipynb pour complier l’Infinite Mixture Model sur des données synthétiques
* Une classe omm.py contenant les fonctions relatives à l’algorithme Infinite Mixture Model
* Une classe utils.py contenant les fonctions qui permettent de générer des données synthétiques, d’initialiser la matrice theta et de formater les données de films
* Un jeu de données contenant les catégories des films (matrice Z) clusters_matrix.csv
* Un jeu de données contenant les données sur films (matrice X) binary_data_matrix.csv

Pour lancer l’algorithme et visualiser les résultats:
* Sélectionner un des trois notebooks mentionnés ci-dessus
* Entrer le nombre d’itération et les dimensions souhaitées (pour les données synthétiques)
* Compiler le code
