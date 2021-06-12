from typing import List, Dict, Tuple
from operator import attrgetter
import numpy as np
from numpy.core.numeric import NaN, count_nonzero
import pandas as pd
from pandas.core.frame import DataFrame
import random as rand

class EventModel:
    def __init__(self, slots_nb: int, persons: List[str], activities: List[str], max_parallel=10) -> None:
        """Construit l'EventModel, il faut ensuite remplir ses informations internes

        Args:
            persons (List[str]): Liste de noms (uniques) des personnes de l'evenement
            slots_nb (int): Nombre de slots de temps disponibles
            game_list (List[str]): Liste des noms (uniques) des jeux disponibles
            max_parallel (int): nombre d'activitées maximum en simultané sur un slot
        """
        
        self.persons: List[str] = persons
        # self.slots: DataFrame = DataFrame(index=range(max_parallel), columns=range(slots_nb))
        self.slots: DataFrame = DataFrame(index=range(slots_nb), columns=['date', 'heure'])
        self.activities: DataFrame = DataFrame(index=activities, columns=['org', 'max_people', 'min_people', 'min_preference'])
        self.preferences: DataFrame = DataFrame(index=self.persons, columns=self.activities.index)

        self.max_parallel = max_parallel
        self.max_preference = 1000

    def get_games_preference_score(self, games: List[str], exclude_players=[]):
        # On exclu les jeux dont un des exclus est GM
        excluded_games = self.activities[self.activities['org'].isin(exclude_players)].index
        games_serie = pd.Series(games)
        games = games_serie[~games_serie.isin(excluded_games)].tolist()
        # On exclu les mises des joueurs exclu
        # print("exclude_players=", exclude_players)
        # print(self.preferences[~self.preferences.index.isin(exclude_players)][games])
        return self.preferences[~self.preferences.index.isin(exclude_players)][games].sum(axis=0)

    def get_best_players(self, game, exclude_players=[]):
        # Serie des joueurs ayant le plus misé sur le jeu
        df = self.preferences[~self.preferences.index.isin(exclude_players)]
        best_players = df[df[game] == df[game].max()][game]
        # Serie si pour chacun des joueur c'est sa meilleure mise
        best_games=(self.preferences[self.preferences.index.isin(best_players.index)].idxmax(axis=1) == game)
      
        # On merge les 2 pour calculer eux pour qui c'est la meilleure mise.
        df_best_players = DataFrame({'preference': best_players, 'best_game': best_games})
        best_player_serie = df_best_players[df_best_players['best_game']]

        best_players = []
        if not best_player_serie.empty:
            best_players = best_player_serie.index.tolist()
            if len(best_players)>1:
                rand.shuffle(best_players)

        return best_players
        

    def get_best_players_in(self, games, exclude_players=[]) -> List[str]:
        """On prends parmis les 2 jeux la mise la plus forte et on l'applique

        Args:
            games (List[str]): jeux à intégrer dans le calcul
            exclude_players (List[str], optional): liste de joueurs à écarter du calcul. Defaults to [].

        Returns:
            List[str]: [description]
        """
        df = self.preferences[~self.preferences.index.isin(exclude_players)][games]
        max_value = df.max().max()

        players_df = df[df[:] == max_value].dropna(how="all", inplace=False)
        players = players_df.index.tolist()
        if len(players)>0:
            rand.shuffle(players)

        for i, p in enumerate(players):
            players[i] = (p, players_df.loc[p,:].idxmax())

        return players

    def from_csv(self, filename_slots, filename_activities, filename_preferences):
        self.preferences: DataFrame = pd.read_csv(filename_preferences, index_col=0)
        self.activities: DataFrame = pd.read_csv(filename_activities, index_col=0)
        self.slots: DataFrame = pd.read_csv(filename_slots, index_col=0)

        persons_list = self.preferences.index.values.tolist()
        if persons_list != self.persons:
            print("ERROR : not same persons", persons_list, "and", self.persons)

    def to_csv(self, filename):
        filename_arr: List[str] = filename.split('.')
        ext = '.'+filename_arr[-1]
        base = '.'.join(filename_arr[:-1])
        
        self.preferences.to_csv(base+"_preferences"+ext)
        self.activities.to_csv(base+"_activities"+ext)
        self.slots.to_csv(base+"_slots"+ext)

    def clean_preferences(self):
        self.preferences.fillna(0, inplace=True)

        test_negative = (self.preferences < 0).any().any()
        if test_negative:
            
            print("ERROR : a preference is not positive")
            # print(self.preferences)
            pers_test = self.preferences[self.preferences < 0].any(axis=1)
            game_test = self.preferences[self.preferences < 0].any(axis=0)
            print(self.preferences.loc[pers_test.values, game_test.values])

            return False
        
        test_too_much = (self.preferences.sum(axis=1) > self.max_preference).any()
        if test_too_much:
            print("ERROR : a user has spent too munch")
            pers_test = self.preferences.sum(axis=1) > self.max_preference
            print(pd.DataFrame({'total': self.preferences.sum(axis=1)}))

            return False

        if not (self.preferences[self.preferences.columns] % 1  == 0).all().all():
            print("ERROR : some values are not integers")
            return False

        return True

class SlotInstance:
    def __init__(self, event_model: EventModel) -> None:
        self.event_model: EventModel = event_model
        self.person_states: DataFrame(index=self.model.preferences.index, columns=['available'])

        # init
        self.person_states['available'] = True


class EventInstance:
    def __init__(self, model: EventModel) -> None:
        self.model: EventModel = model

        # self.slots: List[SlotInstance] = []
        self.games_states: DataFrame = DataFrame(index=self.model.activities.index, columns=['planned', 'available', 'score'])
        self.game_slots_states: DataFrame = DataFrame(index=self.model.slots.index, columns=['full_people', 'full_activity', 'index_to_fill', 'available_players_nb_min', 'available_players_nb_max'])

        # out
        self.plan_activities: DataFrame = DataFrame(index=range(self.model.max_parallel), columns=self.model.slots.index)
        self.plan_persons: DataFrame = DataFrame(index=self.model.persons, columns=self.model.slots.index)

        # init 
        self.init_games_states()
        self.init_game_slots_states()

    def init_games_states(self):
        self.games_states.loc[:,'planned'] = False
        self.games_states.loc[:,'available'] = True
        self.games_states.loc[:,'score'] = 0

    def init_game_slots_states(self):
        self.game_slots_states.loc[:,'full_people'] = False
        self.game_slots_states.loc[:,'full_activity'] = False
        self.game_slots_states.loc[:,'index_to_fill'] = 0
        self.game_slots_states.loc[:,'available_players_nb_min'] = len(self.model.persons)
        self.game_slots_states.loc[:,'available_players_nb_max'] = len(self.model.persons)
        # print(self.game_slots_states)

    def get_nb_players_left(self, slot, game):
        """ On recupere le nombre de personne restantes maximum qu'on peut mettre sur une table sans invalider
        les autres par manque de joueurs.

        Args:
            slot (int): [description]
            game (str): [description]

        Returns:
            [type]: [description]
        """
        # On recupere la liste des jeux du slot (sauf game)
        other_games = self.plan_activities[slot][self.plan_activities[slot] != game][self.plan_activities[slot].notnull()].tolist()
        nb_people_already_in_game = self.plan_persons[slot][self.plan_persons[slot] == game].count()
        if len(other_games)>0:
            min_people_other_games = self.model.activities['min_people'][other_games].sum()
            return len(self.model.persons) - min_people_other_games - nb_people_already_in_game
        else:
            return len(self.model.persons)

    def get_available_games(self):
        return self.games_states[self.games_states['available'] == True].index.tolist()

    def update_game_availablility(self, unavailable_players):
        # On retire les jeux des GM déjà non disponibles
        games_unavailable_players = self.model.activities[self.model.activities['org'].isin(unavailable_players)].index.tolist()
        
        # TODO il faudrait un objet slot qui possede son gamestate pour garder les infos et pas les recalculer
        # On recalcule les jeux non disponibles a partir du planning
        games_unavailable_planning = self.plan_activities.stack().unique()
        # print("games_unavailable_planning=", games_unavailable_planning)

        self.games_states['available'] = True
        self.games_states.loc[games_unavailable_players, 'available'] = False
        self.games_states.loc[games_unavailable_planning, 'available'] = False

    def add_game_to_slot(self, game, slot):
        # add activity to plan
        self.plan_activities.loc[self.game_slots_states.loc[slot,'index_to_fill'], slot] = game
        self.game_slots_states.loc[slot,'index_to_fill'] += 1
        
        # add gm to pla persons
        gm = self.model.activities.loc[game,'org']
        self.plan_persons.loc[gm, slot] = game

        # game not available
        self.games_states.loc[game, 'available'] = False


    def get_best_game(self, slot, by='score') -> str:
        # On enleve les jeux qui ont un nombre de joueur minimum trop élevé pour rentrer dans le slot
        nb_max_players_left_in_slot = self.get_theo_nb_players_left_in_slot(slot, by='min_people')

        activities_filtered = self.model.activities[nb_max_players_left_in_slot > self.model.activities['min_people']].index.tolist()
        # print("filtered by min people:", nb_max_players_left_in_slot, activities_filtered)
        if len(activities_filtered) == 0:
            return NaN

        best_game = self.games_states.loc[activities_filtered, 'score'].idxmax()
        
        return best_game

    def get_games_sorted(self, games, by='score') -> List[str]:
        return self.games_states.loc[games,:].sort_values(by=by, axis=0, ascending=False, inplace=False).index.tolist() 

    def update_games_score(self, games, exclude_players=[]) -> None:
        self.games_states.loc[:,'score'] = self.model.get_games_preference_score(games, exclude_players=exclude_players)

    def get_unavailable_players(self, slot: int) -> List[str]:
        return self.plan_persons[self.plan_persons[slot].notnull()][slot].index.tolist()

    def to_csv(self, filename):
        filename_arr: List[str] = filename.split('.')
        ext = '.'+filename_arr[-1]
        base = '.'.join(filename_arr[:-1])
        
        self.plan_activities.to_csv(base+"_plan_activities"+ext)
        self.plan_persons.to_csv(base+"_plan_persons"+ext)

    def get_games_from_slot(self, slot):
        return self.plan_activities[slot][self.plan_activities[slot].notnull()].tolist()

    def are_game_slots_full_activities(self):
        res = (
            (self.game_slots_states['full_activity'].all()) # Vérifie si les slots sont plein (plus de joueurs dispo)
            or (not self.games_states['available'].any()) # Vérifie si l'état interne indique que c'est full
            or (not self.plan_activities.isnull().any().any()) # Vérifie si il reste des jeux de disponibles
        )
        return res

    def are_game_slot_full_persons(self, slot):
        games = self.get_games_from_slot(slot)
        nb_persons_busy = self.plan_persons[slot].count()

        print("nb_persons_busy=", nb_persons_busy)
        print("max_people_in_games=", self.model.activities[self.model.activities.index.isin(games)]['max_people'].sum() + len(games))
        
        res = (
            (nb_persons_busy == len(self.model.persons)) # Vérifie si il reste des gens de disponibles
            or (
                nb_persons_busy 
                >= self.model.activities[self.model.activities.index.isin(games)]['max_people'].sum() + len(games)
            ) # Il n'y a plus de place sur les jeux
        )

        return res


    def get_theo_nb_players_left_in_slot(self, slot, by):
        activities_in_slot = self.plan_activities[~self.plan_activities[slot].isnull()][slot].values
        nb_players_in_slot = self.model.activities.loc[activities_in_slot, by].sum() + len(activities_in_slot)
        nb_available_players = len(self.model.persons)
        # print("nb_available_players in slot", slot, "=", nb_available_players)
        # print("nb_players_in_slot", slot, "=", nb_players_in_slot)
        return nb_available_players - nb_players_in_slot        

    def are_activities_in_slot_sufficient(self, slot):
        # df des nombre de joueur restant
        nb_min_players_left_in_slot = self.get_theo_nb_players_left_in_slot(slot, by='max_people')
        # print(slot, "->", nb_min_players_left_in_slot)
        return nb_min_players_left_in_slot <= 0

    def set_slot_full_with_activities(self, slot):
        self.game_slots_states.loc[slot, 'full_activity'] = True


### test
if __name__=="__main__":
    model = EventModel(6, 
        ["Alice", "Bob", "Tara", "Leo", "Hans", "Uri", "Lara", "Kenny"], 
        ["toto1", "toto2", "toto3", "toto4", "toto5", "toto6", "toto7", "toto8", "toto9"],
        max_parallel=4
    )

    model.from_csv(
        "in_slots.csv",
        "in_activities.csv",
        "in_preferences.csv" 
    )
    model.to_csv("out2.csv")

    instance = EventInstance(model)
    instance.to_csv("out2.csv")

