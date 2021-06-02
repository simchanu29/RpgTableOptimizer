from typing import List, Dict
from operator import attrgetter
import numpy as np
from collections import OrderedDict
import pandas as pd

class AlphaDictValue:
    def __init__(self, name):
        self.name = name
        self.id = None


class AlphaDict(dict):
    def __init__(self, *args, **kwds) -> None:

        if len(args)>0 and isinstance(args[0], list):
            if len(args[0])>0 and isinstance(args[0][0], AlphaDictValue):
                for elt in args[0]:
                    self.append(elt)
                args = args[1:]

        super().__init__(*args, **kwds)
        self.__data_keys = []
        self.update_keys()
        # self.new_key = False

    def sort_keys(self):
        self.__data_keys = sorted(self.keys()) 

    def update_keys(self):
        # if self.new_key:
        self.sort_keys()
        for i, key in enumerate(self.__data_keys):
            super().__getitem__(key).id = i
            # self.new_key = False

    def __setitem__(self, key, value: AlphaDictValue):
        if isinstance(key, int):
            super().__setitem__(self.__data_keys[key], value)
        else:
            super().__setitem__(key, value)

        # self.new_key = True
        self.update_keys()

    def __getitem__(self, key):
        if isinstance(key, int):
            return super().__getitem__(self.__data_keys[key])
        else:
            return super().__getitem__(key)

    def get_as_list(self):
        return [value for key, value in sorted(self.items())]

    def append(self, value: AlphaDictValue):
        super().__setitem__(value.name, value)
        # self.new_key = True
        self.update_keys()

    def pop(self, key):
        if isinstance(key, int):
            super().pop(self.__data_keys[key])
        else:
            super().pop(key)

        self.update_keys()

    def copy(self):
        return AlphaDict(super().copy())


class Person(AlphaDictValue):
    def __init__(self, name: str):
        super().__init__(name)

        self.preferences: AlphaDict = AlphaDict()
        self.happyness = 0

    def set_preference(self, game, amount: float):
        if game.gm.name != self.name:
            self.preferences[game.name] = Preference(self, game, amount)
        else:
            print("WARNING : gm {} trying to bet on his game {}".format(self, game))

    def get_max_preference(self):
        if len(self.preferences) > 0:
            return max(self.preferences.values(), key=attrgetter('amount'))
        else:
            return None

    def apply_preference(self, game_name: str):
        self.preferences[game_name].apply()

    def __repr__(self):
        return self.name


class GameType(AlphaDictValue):
    def __init__(self, name):
        super().__init__(name)

    def __repr__(self):
        return self.name


class Game(AlphaDictValue):
    def __init__(self, name: str, gametype: GameType, gm: Person,
                 min_players: int = 3, max_players: int = 6, minimum_wage: float = 0):
        super().__init__(name)

        # Definition
        self.game_type: GameType = gametype
        self.gm: Person = gm
        self.players: AlphaDict = AlphaDict()
        self.min_players: int = min_players
        self.max_players: int = max_players
        self.minimum_wage: float = minimum_wage

        # Runtime
        self.planned = False
        self.score = 0

    def __repr__(self):
        return self.name

    def is_none(self):
        return self.game_type is None


class Preference(AlphaDictValue):
    def __init__(self, person: Person, game: Game, amount: float):
        super().__init__(person.name+"_"+game.name)

        # Definition
        self.person = person
        self.game: Game = game
        self.amount = amount

        # Runtime
        self.used = False

    def apply(self):
        # Applique la mise
        self.used = True

        # Ajoute le joueur au jeu
        self.game.players.append(self.person)

    def __repr__(self):
        return "({}, {})".format(self.game, self.amount)


class GameSlot(AlphaDictValue):
    """
    Emplacement temporel dans lequel on peut mettre des jeux plannifiés.

    Peut-être qu'il faudra faire un game_slot definition
    """
    def __init__(self, date, time, players: AlphaDict, max_parallel: int = 1):
        super().__init__(str(date)+"_"+str(time))
        self.date = date
        self.time = time  # aprem ou soir
        self.games: AlphaDict = AlphaDict()
        self.max_parallel: int = max_parallel # not used yet

        self.players: AlphaDict = players

        self.available_players_nb_min: int = len(self.players)
        self.available_players_nb_max: int = len(self.players)
        self.is_full = False

    def add_game(self, game: Game):
        # Attribute slot
        self.games[game.name] = game

        # Remove game from available
        game.planned = True

        # Remove players from player pool min
        self.available_players_nb_max -= game.min_players+1 # players + GM
        self.available_players_nb_min -= game.max_players+1 # players + GM

        self.is_full = (self.available_players_nb_min <= 0)

    def get_unavailable_players(self):
        unavailable_players = []
        for _, game in self.games.items():
            unavailable_players += [player.name for player in game.players.values()]+[game.gm]
        return unavailable_players

    def contain_enough_players_for(self, game: Game):
        return game.min_players < self.available_players_nb_max

    def remove_game(self, game):
        # Attribute slot
        self.games.pop(game.name)

        # Remove game from available
        game.planned = False

        # Remove players from player pool min
        self.available_players_nb_max += game.min_players+1 # players + GM
        self.available_players_nb_min += game.max_players+1 # players + GM

        self.is_full = (self.available_players_nb_min <= 0)

    def __repr__(self):
        return "({}, {}): ".format(self.date, self.time)+str(self.games)


class Event:
    def __init__(self, persons: AlphaDict, game_slots: AlphaDict, games: AlphaDict, is_copy=False):
        self.persons: AlphaDict[str, Person] = persons
        self.game_slots: AlphaDict = game_slots
        
        self.available_games: AlphaDict = games
        self.game_none = Game(None, None, None)

        self.is_copy = is_copy

    def get_random_gameorder(self) -> List[int]:
        """Renvoie une liste aléatoire de jeux unique (pas de id 0)

        Returns:
            List[int]: liste aléatoire d'indice faisant référence à des jeux. l'activité rien faire a été enlevée.
        """

        # On enleve le rien
        game_arr_id = self.available_games.get_as_list()[1:]

        np.random.shuffle(game_arr_id)
        return game_arr_id

    def copy(self):
        return Event(self.persons.copy(), self.game_slots.copy(), self.available_games.copy(), is_copy=True)

    def get_unplanned_games(self):
        games_list = []
        for _, game in self.available_games.items():
            game: Game

            if game.planned:
                continue

            # Il faut vérifier si le gm de ce jeu n'est pas déjà sur une autre table
            game_has_gm = True
            for _, game_other in self.available_games.items():
                game_other: Game
                if game_other != game and game_other.planned and (game.gm is not None):
                    game_has_gm = game_has_gm and not (game.gm.name in game_other.players)

            if game_has_gm:
                games_list.append(game)

        return games_list

    def get_game_preference_score(self, game: Game, exclude_players: List[str]=[]):
        """Sum all the preferences of players on this game

        Args:
            exclude_players (List[Person], optional): [description]. Defaults to [].

        Returns:
            float: preference score
        """
        preference_score: float = 0

        # sum on every person
        for p_key in self.persons:
            # Exclude players from sum
            if self.persons[p_key].name in exclude_players:
                continue

            # Sum on preferences of 1 person
            for pref_key in self.persons[p_key].preferences:
                if self.persons[p_key].preferences[pref_key].game.name == game.name:
                    preference_score += self.persons[p_key].preferences[pref_key].amount

        return preference_score

    def are_game_slots_full(self):
        result = True
        for game_slot in self.game_slots:
            result = result and self.game_slots[game_slot].is_full
        return result

    def to_csv_game(self, filename):
        df_games = pd.DataFrame(index=range(10), columns=self.game_slots.keys())
        for _, game_slot in self.game_slots.items():
            for i, game in enumerate(game_slot.games):
                df_games.loc[i, game_slot.name] = game + " ({})".format(len(game_slot.games[game].players)+1)
        df_games.to_csv(filename)

    def to_csv_person(self, filename):
        df_persons = pd.DataFrame(index=self.persons.keys(), columns=self.game_slots.keys())
        for _, game_slot in self.game_slots.items():
            for _, game in game_slot.games.items():
                df_persons.loc[game.gm.name, game_slot.name] = game.name + " (org)"
                for _, person in game.players.items():
                    df_persons.loc[person, game_slot.name] = game.name
        df_persons.to_csv(filename)

    def to_csv(self, filename):
        self.to_csv_game(filename[:-4]+"_games.csv")
        self.to_csv_person(filename[:-4]+"_persons.csv")

