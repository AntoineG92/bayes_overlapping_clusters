{\rtf1\ansi\ansicpg1252\cocoartf1561\cocoasubrtf200
{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
{\*\listtable{\list\listtemplateid1\listhybrid{\listlevel\levelnfc23\levelnfcn23\leveljc0\leveljcn0\levelfollow0\levelstartat1\levelspace360\levelindent0{\*\levelmarker \{disc\}}{\leveltext\leveltemplateid1\'01\uc0\u8226 ;}{\levelnumbers;}\fi-360\li720\lin720 }{\listname ;}\listid1}
{\list\listtemplateid2\listhybrid{\listlevel\levelnfc23\levelnfcn23\leveljc0\leveljcn0\levelfollow0\levelstartat1\levelspace360\levelindent0{\*\levelmarker \{disc\}}{\leveltext\leveltemplateid101\'01\uc0\u8226 ;}{\levelnumbers;}\fi-360\li720\lin720 }{\listname ;}\listid2}
{\list\listtemplateid3\listhybrid{\listlevel\levelnfc23\levelnfcn23\leveljc0\leveljcn0\levelfollow0\levelstartat1\levelspace360\levelindent0{\*\levelmarker \{disc\}}{\leveltext\leveltemplateid201\'01\uc0\u8226 ;}{\levelnumbers;}\fi-360\li720\lin720 }{\listname ;}\listid3}
{\list\listtemplateid4\listhybrid{\listlevel\levelnfc0\levelnfcn0\leveljc0\leveljcn0\levelfollow0\levelstartat1\levelspace360\levelindent0{\*\levelmarker \{decimal\}.}{\leveltext\leveltemplateid301\'02\'00.;}{\levelnumbers\'01;}\fi-360\li720\lin720 }{\listname ;}\listid4}}
{\*\listoverridetable{\listoverride\listid1\listoverridecount0\ls1}{\listoverride\listid2\listoverridecount0\ls2}{\listoverride\listid3\listoverridecount0\ls3}{\listoverride\listid4\listoverridecount0\ls4}}
\paperw11900\paperh16840\margl1440\margr1440\vieww19200\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # bayes_overlapping_clusters\
Projet ENSAE - Statistiques Bay\'e9siennes - Antoine Grelety, Quentin Spalla, Zakarya Ali\
\
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\b\fs32 \cf0 \ul \ulc0 README: A non parametric approach to modeling Overlapping Clusters\
\

\b0 \ulnone Ce fichier contient:\
\pard\tx220\tx720\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\li720\fi-720\pardirnatural\partightenfactor0
\ls1\ilvl0\cf0 {\listtext	\uc0\u8226 	}Un notebook 
\b IOMM_algo_synthetic_data_K_finite_final.ipynb 
\b0 pour compiler le Mixture Model avec des donn\'e9es synth\'e9tiques
\b \
\ls1\ilvl0
\b0 {\listtext	\uc0\u8226 	}Un notebook
\b  OMM_movies_vf.ipynb
\b0  pour compiler le Mixture Model avec la base de donn\'e9es des films\
\pard\tx220\tx720\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\li720\fi-720\pardirnatural\partightenfactor0
\ls2\ilvl0\cf0 {\listtext	\uc0\u8226 	}Une classe 
\b omm.py
\b0  contenant les fonctions relatives \'e0 l\'92algorithme Mixture Model avec K fini\
\pard\tx220\tx720\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\li720\fi-720\pardirnatural\partightenfactor0
\ls3\ilvl0\cf0 {\listtext	\uc0\u8226 	}Un notebook 
\b IOMM_algo_synthetic_data_K_infinite_final.ipynb
\b0  pour complier l\'92Infinite Mixture Model sur des donn\'e9es synth\'e9tiques\
{\listtext	\uc0\u8226 	}Une classe 
\b omm.py
\b0  contenant les fonctions relatives \'e0 l\'92algorithme Infinite Mixture Model\
{\listtext	\uc0\u8226 	}Une classe 
\b utils.py
\b0  contenant les fonctions qui permettent de g\'e9n\'e9rer des donn\'e9es synth\'e9tiques, d\'92initialiser la matrice theta et de formater les donn\'e9es de films\
{\listtext	\uc0\u8226 	}Un jeu de donn\'e9es contenant les cat\'e9gories des films (matrice Z) 
\b clusters_matrix.csv\
\ls3\ilvl0
\b0 {\listtext	\uc0\u8226 	}Un jeu de donn\'e9es contenant les donn\'e9es sur films (matrice X) 
\b binary_data_matrix.csv
\b0 \
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0
\cf0 \
Pour lancer l\'92algorithme et visualiser les r\'e9sultats:\
\pard\tx220\tx720\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\li720\fi-720\pardirnatural\partightenfactor0
\ls4\ilvl0\cf0 {\listtext	1.	}S\'e9lectionner un des trois notebooks mentionn\'e9s ci-dessus\
{\listtext	2.	}Entrer le nombre d\'92it\'e9ration et les dimensions souhait\'e9es (pour les donn\'e9es synth\'e9tiques)\
{\listtext	3.	}Compiler le code\
}