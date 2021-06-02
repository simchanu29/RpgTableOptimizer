import numpy as np
from typing import List, Dict, Tuple
from Models import Event, GameSlot, Game, GameType, Person, Preference, AlphaDict, AlphaDictValue
from operator import attrgetter


class Optimizer:
    def __init__(self) -> None:
        pass

    def optimize(self, event: Event):
        """Fonction d'optimisation a surcharger

        Args:
            event (Event): evenement source a optimiser
        """
        print("ERROR : optimize function not overloaded")

    def compute_happyness(self, event: Event):
        """Calcule le contentement des gens par rapport à leur préférences

        Pour chaque personne, on calcule si sa mise a été prise en compte ou non :
        - Si le jeu est plannifié

        Args:
            event (Event): evenement à auditer
        """
        happyness = 0

        for pers in event.persons.values():
            # pers = Person()
            for pref in pers.preferences.values():
                # pref = Preference()
                # if pref.game.planned:
                #     pers.happyness += pref.amount
                #     happyness += pref.amount
                # else:
                #     pers.happyness -= pref.amount
                #     happyness -= pref.amount
                pass
                

            # if game_slot is not None:
        return happyness


class OptimizerDeterminist(Optimizer):
    def __init__(self) -> None:
        super().__init__()

    def fill_slots_from_preferences_once(self, event: Event, games_sorted_by_score: List[Game]):
        for slot in event.game_slots:
            # Pour chaque 1ère activité du slot, on prends l'activité avec le plus de mise
            slot.add_game(games_sorted_by_score[0])

            # Puis on la retire
            games_sorted_by_score.pop(0)

            # Ensuite on met tout les joueurs dont c'est la mise maximale dessus    
            for p_key in slot.players:
                max_preference = slot.players[p_key].get_max_preference()
                # print(event.game_slots[i], p_key, max_preference)

                # Si il n'y a aucune préférence de renseignée on passe
                if max_preference is None:
                    continue

                # Si la mise maximale est dans le slot alors on l'applique
                if max_preference.game.name in slot.games:
                    # Applique la mise
                    max_preference.apply(slot.players[p_key])

    def fill_slots_from_preferences(self, event: Event):
        """Utilise les préférences des personnes de l'evenement pour mettre en place 
        les activités avec une règle similaire aux enchères.
        """

        while not event.are_game_slots_full():
            # On somme les mise sur une activité et on les classe de manière décroissante
            print("=== LOOP")
            for slot in event.game_slots.values():
                slot: GameSlot

                # Pour chaque slot, on calcule le score de chaque jeu en prenant 
                # en compte les jeux qui ont déjà été validé, ce qui implique
                # que il y a des joueurs en moins pour calculer ce qui est le mieux 
                # pour les slots restants

                # On calcule le score en retirant les joueurs qui ont déjà et leur mise validée            
                # On choppe la liste des noms des joueurs qui ne sont plus dispo sur un slot
                unavailable_players = slot.get_unavailable_players()
                print("slot =", slot, "unavailable_players =",unavailable_players)

                # On met à jour le score pour tous les jeu 
                for game in event.available_games.values():
                    game.score = event.get_game_preference_score(game, exclude_players=unavailable_players)

                # On recupère la liste 
                # Si un MJ est déjà indisponible car sur une autre table déjà programmée sur ce slot il faut que ça soit pris en compte
                games_sorted_by_score = sorted(event.get_unplanned_games(), key=lambda obj: obj.score, reverse=True)
                games_sorted_by_score.remove(event.available_games[0]) # Pour enlever le None, TODO : None nécessaire ?

                print("game list :", games_sorted_by_score)

                # Pour chaque 1ère activité du slot, on prends l'activité avec le plus de mise
                if len(games_sorted_by_score) > 0:
                    # Donc ici on prend le 1er de la liste
                    slot.add_game(games_sorted_by_score[0])
                    print("game chosen: ", games_sorted_by_score[0])
                else:
                    print("No more games available for slot ", slot)
                    slot.is_full = True

                # Ensuite on met tout les joueurs disponibles dont c'est la mise maximale dessus    
                # Pour chaque jeu choisi dans le slot on organise les mises de la plus gande à la plus petite
                for _, game in slot.games.items():
                    game: Game
                    print("Checking preferences for", game.name)

                    players_waiting_list: List[Tuple[Person, Preference]] = []

                    for p_key in slot.players:
                        if p_key in unavailable_players:
                            continue

                        max_preference: Preference = slot.players[p_key].get_max_preference()

                        # Si il n'y a aucune préférence de renseignée on passe
                        if max_preference is None:
                            continue

                        # On ajoute le joueur a une liste d'attente pour valider sa mise
                        if max_preference.game.name is game.name:
                            players_waiting_list.append((slot.players[p_key], max_preference))
                    
                    # On trie cette liste de joueur en fonction de la valeur de leur mise
                    players_waiting_list.sort(key=lambda obj: obj[1].amount, reverse=True)

                    # Tant que la waiting list n'est pas vide ou que la taille minimale de la table
                    # n'a pas été atteinte on remplit la table
                    print("Player waiting list :", players_waiting_list)
                    while len(players_waiting_list)!=0 or len(game.players)>=game.min_players:
                        preference = players_waiting_list[0][1]
                        person = players_waiting_list[0][0]

                        preference.apply()
                        players_waiting_list.pop(0)

                        print(preference, "->", person)
                        


        # print(event.game_slots)
        # print(games_sorted_by_score)
        # print(event.are_game_slots_full())
        # for slot in event.game_slots.values():
        #     print(slot, slot.available_players_nb_min, slot.available_players_nb_max, slot.is_full)

        # Puis on recalcule la somme des mises pour le slot en retirant les joueurs indisponibles
        # Ensuite si il reste de joueurs en config table maximale, on selectionne les autres jeux avec un maximum de mises

        # Ensuite on met tout les joueurs restant dont c'est la mise maximale dessus    

        # Puis on met les joueurs restant sur les tables en fonction de leur mise maximale sur les jeux disponible sur le slot.
        # Pour rester déterministe en cas d'égalité de mise c'est la première (puis décroissant) qui est sélectionnée

    def optimize(self, event: Event):
        self.fill_slots_from_preferences(event)


class OptimizerGenetic(Optimizer):
    """
    L'optimisation se découpe en plusieurs phases
    1. Génération de la génération initiale
    2. Calcul du contentement pour chacun des éléments de la génération
    3. Phase de reproduction pour la géneration suivante
    """
    def __init__(self):
        super().__init__()
        self.rules = []

    def optimize(self, event: Event):
        print(event.get_random_gameorder())
        self.gen_slots(event)

    def generate_events(self):
        """
        Generate a set of events
        """
        pass

    def fill_slots_once(self, event: Event, game_arr_id: List[int]):
        '''
        On privilégie les jeux à beaucoup si c'est possible.
        On considère l'event comme une copie, on considère qu'on peut le modifier
        '''

        # On remplit les gameslot avec des jeux issu du gameorder 
        # Pour chaque game slot 
        for slot in event.game_slots:
            # On vérifie si avec les jeux actuels dans le slots on a encore besoin de personnes
            if slot.is_full:
                print("No game available anymore")
                continue

            # Avec la somme du nombre max de places dans chacun des jeux du slot on dépasse pas le nombre de personnes dispo
            # On essaie donc de remplir ce slot        

            # On remplis le slot i avec le 1er jeu qui remplit la condition de nombre de joueur minimum
            # Pour chaque jeu
            for game_id in game_arr_id:
                game = event.available_games[game_id]

                # Si le jeu est déjà plannifié on passe au suivant
                if game.planned:
                    continue

                # Si on a assez de joueurs pour le jeu (strict car il y a le MJ aussi)
                if slot.contain_enough_players_for(game):
                    # On met le jeu dans le game_slot et on le retire de disponible
                    slot.add_game(game)
                    break

            # Si il y a plus rien à faire alors la boucle ne fera rien et on se
            # retrouvera avec un selected_game_name vide

    def gen_slots(self, event: Event):
        '''
        Génération d'un ensemble de slots remplis par des jeux

        Tant que on peut le faire (qu'il y a besoin de rajouter des jeux)
        On remet une couche de jeu

        Cette fonction considère un event puis les organise. Il est important de noter que cet event
        '''

        if not event.is_copy:
            print("WARNING : event not a copy")

        # Génération alétaoire d'id
        # C'est l'ordre aleatoire dans lequel les jeux vont être proposés au remplissage
        game_arr_id = event.get_random_gameorder()

        # Copie des game slots pour éviter de modifier l'event
        tmp_game_slots = event.game_slots.copy()
        tmp_available_games = event.available_games.copy()

        # TODO While les slots ne sont pas tous plein
        # On fait une passe individuelle sur chaque slot pour y rajouter un jeu

        while not event.are_game_slots_full():
            self.fill_slots_once(event, game_arr_id)
            pass

        # On a remplis tout les slots avec quelques jeux
        # On effectue la derniere etape
        # Pour chaque slot
        for i, _ in enumerate(event.game_slots):
           # - Pour l'instant on fait rien mais on pourrais imaginer de mettre 
           # un paramètre aléatoire par slot qui permet de choisir entre avoir des grosses parties 
           # et avoir des petites parties
           # - OU sinon de mettre un paramétrage accessible aux MJ pour qu'on tente d'optimiser le nombre de joueurs 
           # idéaux afin que 
           # - OU sinon on laisse ça a l'optimisation des mises qui pourra rajouter des jeux ?
           pass

    def set_rule(self, rule):
        self.rules.append(rule)

    def fill_slots(self):
        pass


class MyTestBench:
    def __init__(self) -> None:
        self.persons = AlphaDict([
            Person("Alice"),
            Person("Bob"),
            Person("Tara"),
            Person("Leo"),
            Person("Hans"),
            Person("Uri"),
            Person("Lara"),
            Person("Kenny")
        ])

        self.game_slots = AlphaDict([
            GameSlot(0, 'aprem', self.persons.copy(), max_parallel=3),
            GameSlot(0, 'soir' , self.persons.copy(), max_parallel=3),
            GameSlot(1, 'aprem', self.persons.copy(), max_parallel=3),
            GameSlot(1, 'soir' , self.persons.copy(), max_parallel=3)
        ])

        self.games = AlphaDict([
            Game("_None", None, None),
            Game("toto1", GameType("toto"), self.persons["Alice"]),
            Game("toto2", GameType("toto"), self.persons["Tara"]),
            Game("toto3", GameType("toto"), self.persons["Leo"]),
            Game("toto4", GameType("toto"), self.persons["Leo"]),
            Game("toto5", GameType("toto"), self.persons["Alice"]),
            Game("toto6", GameType("toto"), self.persons["Alice"]),
            Game("toto7", GameType("toto"), self.persons["Leo"]),
            Game("toto8", GameType("toto"), self.persons["Lara"]),
            Game("toto9", GameType("toto"), self.persons["Alice"]),
        ])

        # Preferences
        # TODO faut passer ça sur une matrice
        self.persons['Alice'].set_preference(self.games["toto2"], 300)
        self.persons['Hans'].set_preference(self.games["toto2"], 300)
        self.persons['Hans'].set_preference(self.games["toto3"], 301)
        self.persons['Hans'].set_preference(self.games["toto4"], 300)
        self.persons['Leo'].set_preference(self.games["toto4"], 300)
        self.persons['Leo'].set_preference(self.games["toto5"], 101)
        self.persons['Leo'].set_preference(self.games["toto6"], 300)
        self.persons['Leo'].set_preference(self.games["toto1"], 299)
        self.persons['Leo'].set_preference(self.games["toto9"], 296)
        self.persons['Lara'].set_preference(self.games["toto4"], 188)
        self.persons['Lara'].set_preference(self.games["toto4"], 302)
        self.persons['Lara'].set_preference(self.games["toto4"], 200)

        self.event = Event(self.persons, self.game_slots, self.games)


if __name__ == "__main__":
    # Init event
    test = MyTestBench()
    optimizer = OptimizerDeterminist()
    optimizer.optimize(test.event)
    print(test.event.game_slots)
    
    test.event.to_csv("out.csv")

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

