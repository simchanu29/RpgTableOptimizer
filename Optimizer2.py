import numpy as np
from typing import List, Dict, Tuple
from Models2 import EventModel, EventInstance, SlotInstance
from operator import attrgetter
import pandas as pd
from pandas.core.frame import DataFrame


class Optimizer:
    def __init__(self) -> None:
        pass

    def optimize(self, event: EventModel):
        """Fonction d'optimisation a surcharger

        Args:
            event (Event): evenement source a optimiser
        """
        print("ERROR : optimize function not overloaded")

    def compute_happyness(self, event: EventInstance):
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

    def fill_slots_from_preferences_once(self, event: EventInstance, games_sorted_by_score: List):
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

    def fill_slots_from_preferences(self, event: EventInstance):
        """Utilise les préférences des personnes de l'evenement pour mettre en place 
        les activités avec une règle similaire aux enchères.
        """

        # Remplissage des activités dans les slots
        while not event.are_game_slots_full_activities():
            # On somme les mise sur une activité et on les classe de manière décroissante
            print("=== FILL ACTIVITY LOOP")
            for slot in event.plan_activities:
                # Joueurs qui sur ce slot sont déjà dans un jeu ou sont le gm d'un autre sont à enlever du choix
                unavailable_players = event.get_unavailable_players(slot)

                # Pour les joueurs unavailable on met leur jeu en unavailable
                event.update_game_availablility(unavailable_players)
                games = event.get_available_games()

                # Somme les mises du modele selon les arguments sur les lignes et les place dans games_states
                event.update_games_score(games, exclude_players=unavailable_players)

                # On recupere la liste ordonnee des jeux en fonction de leur score
                best_game = event.get_best_game(slot, by='score')

                if isinstance(best_game, str):
                    print("Add game", best_game, "to plan in slot", slot)
                    event.add_game_to_slot(best_game, slot)
                    # Si le dernier jeu peut mettre tout le monde sur une seule activité alors on considere le slot full aussi
                    if event.are_activities_in_slot_sufficient(slot):
                        print("Sufficient, end slot", slot)
                        event.set_slot_full_with_activities(slot)       

                    # Pour optimiser on met le joueur le plus intéressé par la table dessus (en cas d'égalité il y a de l'aleatoire)
                    best_players = event.model.get_best_players(best_game, exclude_players=unavailable_players)
                    if len(best_players)>0:
                        print("best_players=",best_players)

                        # recupere le nombre de joueur maximum que peut acceuillir la table
                        max_players_nb = event.model.activities['min_people'][best_game]
                        for i in range(min(max_players_nb, len(best_players))):
                            event.plan_persons.loc[best_players[i], slot] = best_game

                    else: 
                        print("No best player available : bet more on another table")                     
                else: 
                    # On ne peut plus trouver de jeu pour ce slot : il est full
                    event.set_slot_full_with_activities(slot)
                    print("No more games for slot", slot)
        
        # Remplissage des gens dans les activités
        
        # Pour chaque slot
        for slot in event.plan_activities:
            while not event.are_game_slot_full_persons(slot):
                print("=== FILL PEOPLE LOOP")
                # Pour chaque table dans le slot
                unavailable_players = event.get_unavailable_players(slot)
                print("unavailable_players=", unavailable_players)

                # On prends la personne avec la plus grande mise parmis les 2 tables du slot
                games = event.plan_activities[slot][event.plan_activities[slot].notnull()].tolist()
                best_players = event.model.get_best_players_in(games, exclude_players=unavailable_players)

                # Il faut que le nombre max de joueur rajoutable sur la table ne depasse pas
                # (le nombre de joueurs total) - (somme des min des autres tables)
                for p in best_players:
                    nb_players_left = event.get_nb_players_left(slot, p[1])
                    if(nb_players_left>0):
                        event.plan_persons.loc[p[0], slot] = p[1]
                



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

    def optimize(self, event: EventModel):
        instance = EventInstance(event)
        
        self.fill_slots_from_preferences(instance)

        return instance


if __name__ == "__main__":
    # Init event
    model = EventModel(3, 
        ["Alice", "Bob", "Tara", "Leo", "Hans", "Uri", "Lara", "Kenny"], 
        ["toto1", "toto2", "toto3", "toto4", "toto5", "toto6", "toto7", "toto8", "toto9"],
        max_parallel=4
    )

    model.from_csv(
        "in_slots.csv",
        "in_activities.csv",
        "in_preferences.csv" 
    )

    optimizer = OptimizerDeterminist()
    event = optimizer.optimize(model)
    
    event.to_csv("out2.csv")
