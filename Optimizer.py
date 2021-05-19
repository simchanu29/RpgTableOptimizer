import numpy as np
from typing import List, Dict

class Person:
    def __init__(self, name):
        self.name: str = name
        self.preferences: Dict[str, Preference] = {}

    def set_preference(self, game, amount):
        self.preferences[game.name] = Preference(game, amount)

    def __repr__(self):
        return self.name

class RPG:
    def __init__(self, name):
        self.name: str = name

    def __repr__(self):
        return self.name

class Preference:
    def __init__(self, game, amount):
        self.game: Game = game
        self.amount = amount

    def __repr__(self):
        return "({}, {})".format(self.game, self.amount)

class Game:
    def __init__(self, gameid, name, rpg, gm, min_players=3, max_players=6):
        self.name: str = name
        self.rpg: RPG = rpg
        self.gm: Person = gm
        self.players: List[Person] = None
        self.id: int = gameid
        self.min_players: int = min_players
        self.max_players: int = max_players

    def __repr__(self):
        return self.name

class GameSlot:
    def __init__(self, date, time, max_parallel=1):
        self.date = date
        self.time = time # aprem ou soir
        self.games: Dict[str, Game] = {}
        self.max_parallel: int = max_parallel

    def __repr__(self):
        return str(self.games)

class Event:
    def __init__(self, persons, game_slots: List[GameSlot], games):
        self.persons: Dict[str, Person] = persons
        self.game_slots: List[GameSlot] = game_slots
        
        self.available_games: List[Game] = games
        self.game_none = Game(0, None, None, None)


    def get_random_gameorder(self) -> List[int]:
        '''
        Renvoie une liste aléatoire de jeux unique (pas de id 0)
        '''

        # On enleve le rien
        game_arr_id = list(range(len(self.available_games)))[1:]

        np.random.shuffle(game_arr_id)
        return game_arr_id


class Optimizer:
    '''
    L'optimisation se découpe en plusieurs phases
    1. Génération de la génération initiale
    2. Calcul du contentement pour chacun des éléments de la génération
    3. Phase de reproduction pour la géneration suivante
    '''
    def __init__(self):
        self.rules = []

    def optimize(self, event: Event):
        print(event.get_random_gameorder())
        # self.gen_slots(event)

    def gen_slots(self, event: Event):
        '''
        Génération d'un ensemble de slots remplis par des jeux

        Tant que on peut le faire (qu'il y a besoin de rajouter des jeux)
        On remet une couche de jeu
        '''

        # Génération alétaoire d'id
        game_arr_id = event.get_random_gameorder()

        # Copie des game slots pour éviter de modifier l'event
        tmp_game_slots = event.game_slots.copy()
        tmp_available_games = event.available_games.copy()

        # While TODO

        # On remplit les gameslot avec des jeux issu du gameorder 
        for i, _ in enumerate(tmp_game_slots):
            players_pool = len(event.persons)
            selected_game_name = ""

            # On remplis le slot i avec le 1er jeu qui remplis la condition de nombre de joueur minimum
            for j, _ in enumerate(event.available_games):
                # Si il y a autre chose que rien à faire
                if len(tmp_available_games) > 1:
                    game_id = game_arr_id[j]

                    # Strict car il y a le MJ aussi
                    if tmp_available_games[game_id].min_players < players_pool:
                        # Save game name
                        selected_game_name = tmp_available_games[game_id].name

                        # Attribute slot
                        tmp_game_slots[i].games[selected_game_name] = tmp_available_games[game_id]
                        
                        # Remove game from available
                        tmp_available_games.pop(game_id)

                        # Remove players from player pool
                        # players_pool -= tmp_available_games[game_id].min_players+1 # players + GM
                        break
                # Si il n'y a que rien à faire
                else:
                    # Activite = rien
                    game_id = 0
                    selected_game_name = tmp_available_games[game_id].name
                    tmp_game_slots[i].games[selected_game_name] = tmp_available_games[game_id]
                    break


        # On a remplis tout les slots avec  jeu
        for i, _ in enumerate(tmp_game_slots):
            # On regarde si on peut rajouter un jeu au slot i
            if tmp_game_slots[i].games[selected_game_name].max_players > players_pool:
                # On peut étendre le jeu pour contenir tout le monde
                # On refera un second passage si il reste des jeux à programmer
                pass
            else:
                # Le jeu n'est pas assez grand pour contenir tout le monde
                # Il faut nécessairement rajouter des activités pour que tout le monde puisse en avoir une
                pass



            
            
            # Strict car il y a le MJ aussi
            if players_pool > tmp_game_slots[0].min_players:
                pass

        # print(tmp_game_slots)


    def compute_happyness(self, event: Event):
        # for game_slot in event.game_slots:
            # if game_slot is not None:
        pass

    def set_rule(self, rule):
        self.rules.append(rule)

    def fill_slots(self):
        pass

class MyTestBench:
    def __init__(self) -> None:
        self.persons = {
            "Alice": Person("Alice"),
            "Bob": Person("Bob"),
            "Tara": Person("Tara"),
            "Leo": Person("Leo"),
            "Hans": Person("Hans"),
            "Uri" : Person("Uri"),
            "Lara": Person("Lara"),
        }

        self.game_slots = [
            GameSlot(0, 'aprem', max_parallel=3),
            GameSlot(0, 'soir' , max_parallel=3),
            GameSlot(1, 'aprem', max_parallel=3),
            GameSlot(1, 'soir' , max_parallel=3)
        ]

        self.games = [
            Game(0, "None", None, None),
            Game(1, "toto1", RPG("toto"), self.persons["Alice"]),
            Game(2, "toto2", RPG("toto"), self.persons["Tara"]),
            Game(3, "toto3", RPG("toto"), self.persons["Leo"]),
            Game(4, "toto4", RPG("toto"), self.persons["Leo"]),
            Game(5, "toto5", RPG("toto"), self.persons["Alice"]),
            Game(6, "toto6", RPG("toto"), self.persons["Alice"]),        
        ]

        # Preferences
        # TODO faut passer ça sur une matrice
        self.persons['Alice'].set_preference(self.games[1], 300)
        self.persons['Hans'].set_preference(self.games[2], 300)
        self.persons['Hans'].set_preference(self.games[3], 300)
        self.persons['Hans'].set_preference(self.games[4], 300)

        self.event = Event(self.persons, self.game_slots, self.games)

if __name__ == "__main__":
    # Init event
    test = MyTestBench()
    optimizer = Optimizer()
    optimizer.optimize(test.event)

"""
Travailler par matrices
M = matrices des mises (joueur x tables)
V = masque de validation (joueur x tables)

Criteres :
- Validité des mises
- Tableau de l'amour (joueurs)
- Minimiser les tables des MJs (diversité MJ)
- Pas de personne toute seule sur un slot de jeu. 
- Minimiser l'écart de joueur par rapport au nombre de joueurs prévu
- Tables minimum à 3 joueurs + 1 MJ.

Problèmes : 
- Des gens déjà sur des tables.
- Des tables déjà validées
=> Condition dans la génération de la matrice V

Entrées : 
M = mises
nombre n de slots

Generé : 
Matrice slots vs jeux.
Matrice validation mises.

Sorties : 
Matrice slots vs jeux.
Matrice validation mises.

Algorithme génétique

"""

