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

        # Generate dataframe of confirmed activities.
        df_pref = event.model.preferences.copy()

        # Mise validée si 
        # - Le jeu est programmé et si la personne est dans le jeu

        df_happyness = pd.DataFrame()
        for pers in event.model.persons:
            df_pref_pers = df_pref.loc[pers, :]
            games_serie_person_planned = event.plan_persons.loc[pers, :]

            games_serie_person_not_planned = df_pref_pers[~df_pref_pers.index.isin(games_serie_person_planned)].index

            df_pref.loc[pers, games_serie_person_not_planned] = df_pref.loc[pers, games_serie_person_not_planned].multiply(-1, fill_value=0)

        df_happyness = df_pref.sum(axis=1)
        print(df_pref)
        print()
        print(df_happyness)

        return df_happyness.min()


class OptimizerDeterminist(Optimizer):
    def __init__(self) -> None:
        super().__init__()

    def insert_best_game_in_slot(self, event: EventInstance, slot: SlotInstance):
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

    def insert_best_people_in_slot(self, event, slot):        
        # Pour chaque table dans le slot
        unavailable_players = event.get_unavailable_players(slot)
        print("unavailable_players=", unavailable_players)
        
        # unavailable_games = event.get_full_games_from_slot(slot)
        # print("unavailable_games=", unavailable_games)

        # On prends la personne avec la plus grande mise parmis les 2 tables du slot
        games = event.get_games_from_slot(slot)
        best_players = event.model.get_best_players_in(games, exclude_players=unavailable_players)

        # Il faut que le nombre max de joueur rajoutable sur la table ne depasse pas
        # (le nombre de joueurs total) - (somme des min des autres tables)
        for p in best_players:

            # Gestion des mises égales
            if len(p[1])>1:
                # On prend le jeu qui a actuellement le plus de place restante
                games_players_left = [(event.get_nb_players_left_in_slot_for_game(slot, game), game) for game in p[1]]
                p[1] = max(games_players_left, key=lambda x: x[0])[1]
            else:
                p[1] = p[1][0]

            nb_players_left = event.get_nb_players_left_in_slot_for_game(slot, p[1])
            if(nb_players_left>0):
                event.plan_persons.loc[p[0], slot] = p[1]

    def fill_slots_from_preferences(self, event: EventInstance):
        """Utilise les préférences des personnes de l'evenement pour mettre en place 
        les activités avec une règle similaire aux enchères.
        """

        # Remplissage des activités dans les slots
        depth = 0
        while not event.are_game_slots_full_activities():
            # On somme les mise sur une activité et on les classe de manière décroissante
            if depth % 2 == 0:
                slot_list = event.plan_activities.columns.tolist()
            else:
                slot_list = reversed(event.plan_activities.columns.tolist())

            for slot_id in slot_list:
                print("=== FILL ACTIVITY LOOP :", slot_id)
                self.insert_best_game_in_slot(event, slot_id)
            
            depth+=1


        # Remplissage des gens dans les activités
        
        # Pour chaque slot
        for slot in event.plan_activities:
            while not event.are_game_slot_full_persons(slot):
                # if slot == '180':
                #     print("DEBUG")
                print("=== FILL PEOPLE LOOP :", slot)
                self.insert_best_people_in_slot(event, slot)

        # print(event.game_slots)
        # print(games_sorted_by_score)
        # print(event.are_game_slots_full())
        # for slot in event.game_slots.values():
        #     print(slot, slot.available_players_nb_min, slot.available_players_nb_max, slot.is_full)

    def optimize(self, event: EventModel):
        instance = EventInstance(event)
        
        self.fill_slots_from_preferences(instance)

        self.compute_happyness(instance)

        return instance


if __name__ == "__main__":
    # Init event
    model = EventModel(max_parallel=4)

    model.from_csv(
        "in_slots.csv",
        "in_activities.csv",
        "in_preferences.csv" 
    )

    if model.clean_preferences():
        optimizer = OptimizerDeterminist()
        event = optimizer.optimize(model)
        
        event.to_csv("out2.csv")
