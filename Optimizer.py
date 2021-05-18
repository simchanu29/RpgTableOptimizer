import numpy as np

class Person:
    def __init__(self, name):
        self.name = name
        self.preferences = {}

    def set_preference(self, game, amount):
        self.preferences[game.name] = Preference(game, amount)

    def __repr__(self):
        return self.name

class RPG:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

class Preference:
    def __init__(self, game, amount):
        self.game = game
        self.amount = amount

    def __repr__(self):
        return "({}, {})".format(self.game, self.amount)

class Game:
    def __init__(self, gameid, name, rpg, gm):
        self.name = name
        self.rpg = rpg
        self.gm = gm
        self.players = None
        self.id = gameid

    def __repr__(self):
        return self.name

class GameSlot:
    def __init__(self, date, time, max_parallel=1):
        self.date = date
        self.time = time # aprem ou soir
        self.games = {}
        self.max_parallel = max_parallel

    def __repr__(self):
        return str(self.games)

class Event:
    def __init__(self, persons, game_slots, games):
        self.persons = persons
        self.game_slots: list = game_slots
        
        self.available_games = games
        self.game_none = Game(0, None, None, None)


    def get_random_gameorder(self):
        # On enleve le rien
        game_arr_id = list(range(len(self.available_games)))[1:]

        np.random.shuffle(game_arr_id)
        return game_arr_id


class Optimizer:
    def __init__(self):
        self.rules = []

    def optimize(self, event):
        # print(event.get_random_gameorder())
        self.gen_slots(event)

    def gen_slots(self, event):
        game_arr_id = event.get_random_gameorder()

        tmp_game_slots = event.game_slots.copy()

        for i in range(len(tmp_game_slots)):
            if i == 0:
                tmp_game_slots[0] = event.available_games[game_arr_id[0]]
                if 

        print(tmp_game_slots)


    def compute_happyness(self, event: Event):
        # for game_slot in event.game_slots:
            # if game_slot is not None:
        pass

    def set_rule(self, rule):
        self.rules.append(rule)

    def fill_slots(self):
        pass

if __name__ == "__main__":
    # Init event
    persons = {
        "Alice": Person("Alice"),
        "Bob": Person("Bob"),
        "Tara": Person("Tara"),
        "Leo": Person("Leo"),
        "Hans": Person("Hans"),
        "Uri" : Person("Uri"),
        "Lara": Person("Lara"),
    }

    game_slots = [
        GameSlot(0, 'aprem', max_parallel=3),
        GameSlot(0, 'soir' , max_parallel=3),
        GameSlot(1, 'aprem', max_parallel=3),
        GameSlot(1, 'soir' , max_parallel=3)
    ]

    games = [
        Game(0, "None", None, None),
        Game(1, "toto1", RPG("toto"), persons["Alice"]),
        Game(2, "toto2", RPG("toto"), persons["Tara"]),
        Game(3, "toto3", RPG("toto"), persons["Leo"]),
        Game(4, "toto4", RPG("toto"), persons["Leo"]),
        Game(5, "toto5", RPG("toto"), persons["Alice"]),
        Game(6, "toto6", RPG("toto"), persons["Alice"]),        
    ]
    
    event = Event(persons, game_slots, games)
    optimizer = Optimizer()

    # Preferences
    # TODO faut passer ça sur une matrice
    persons['Alice'].set_preference(games[1], 300)
    persons['Hans'].set_preference(games[2], 300)
    persons['Hans'].set_preference(games[3], 300)
    persons['Hans'].set_preference(games[4], 300)

    optimizer.optimize(event)

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

