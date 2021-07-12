# RpgTableOptimizer

Optimisateur de gens sur des actiuvités en tenant en compte leurs préférences sous la forme de mises.

### Dépendances

- pandas
- numpy

### Utilisation

#### En local

Remplir les tableaux csv nommés `in_<quelque chose>.csv` et lancer Optimizer2.py

Le résultat est disponible dans les fichiers nommés `out2_<quelque chose>.csv`

#### En Réseau

Lancer `OptimizerServer.py`. Un serveur va se lancer pour écouter les requêtes externes sur le port 8000.

Pour tester ce serveur on peut utiliser le script `tests/test_client_http.py`. 

### Installation

1. Lancer `OptimizerServer.py` sur un serveur. Le mieux c'est d'en faire un service. Sinon on peut le lancer avec la commande screen.
```
screen -d -m -S OptimizerServer python3 OptimizerServer.py
```
2. Copier coller les scripts GAS dans l'éditeur de script d'une google sheet.

### Fonctionnement

#### Déterministe

L'optimisation déterministe simple fonctionne de la manière suivante : 
1. On définis les activités par slot
  1. Pour chaque slot on choisis un jeu rassemblant le plus de mises pour les gens sur le slot
  2. On valide ensuite toutes les personnes dont la mise maximum est sur ce jeu (dans la limite des places disponibles)
  3. On continue balayer les slots (a chaque fois en inversant le sens de balayage) pour les remplir d'un jeu jusqu'à ce que la somme du maximum de place sur chaque activité dépasse le nombre de gens sur le slot
2. Ensuite on attribue les gens aux activités
  1. Pour chaque slot on prend la personne encore disponible qui a misé le plus sur les activités du slot et on lui valide sa place

#### Astar

WIP

### Known issues
- Quand il n'y a plus assez de personne pour remplir des tables en tenant en compte le nombre de minimum pour constituer une table on a une boucle infinie. 

### TODO list

- Optimisation par algorithme génétique
- Intégration avec google sheet effectuant des requete au programme tournant sur un serveur distant.
- Fonction de coût : 
  - Optimisation de la diversité des joueurs
  - Optimisation de la diversité des MJs
- Fonctionnalité de réservation de table
- Fonctionnalité de réservation de joueur
