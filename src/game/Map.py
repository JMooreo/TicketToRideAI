from typing import Dict

from src.game.enums.TrainColor import TrainColor
from src.game.enums.City import City
from src.game.Destination import Destination
from src.game.Route import Route
from src.game.RouteCost import RouteCost


class USMap:
    def __eq__(self, other):
        return isinstance(other, USMap) and \
            all([r in other.routes for r in self.routes]) and \
            all([d in other.destinations for d in self.destinations])

    def __init__(self):
        self.routes: Dict[Route] = {
            # ([City1, City2], RouteCost(Color, Amount), optional* adjacent route)
            0: Route(0, [City.ATLANTA, City.NASHVILLE], RouteCost(TrainColor.WILD, 1)),
            1: Route(1, [City.VANCOUVER, City.SEATTLE], RouteCost(TrainColor.WILD, 1), 2),
            2: Route(2, [City.VANCOUVER, City.SEATTLE], RouteCost(TrainColor.WILD, 1), 1),  # not a typo, there's two
            3: Route(3, [City.PORTLAND, City.SEATTLE], RouteCost(TrainColor.WILD, 1), 4),
            4: Route(4, [City.PORTLAND, City.SEATTLE], RouteCost(TrainColor.WILD, 1), 3),  # not a typo, there's two
            5: Route(5, [City.DALLAS, City.HOUSTON], RouteCost(TrainColor.WILD, 1), 6),
            6: Route(6, [City.DALLAS, City.HOUSTON], RouteCost(TrainColor.WILD, 1), 5),  # not a typo, there's two
            7: Route(7, [City.OMAHA, City.KANSAS_CITY], RouteCost(TrainColor.WILD, 1), 8),
            8: Route(8, [City.OMAHA, City.KANSAS_CITY], RouteCost(TrainColor.WILD, 1), 7),  # not a typo, there's two
            9: Route(9, [City.DULUTH, City.OMAHA], RouteCost(TrainColor.WILD, 2), 10),
            10: Route(10, [City.DULUTH, City.OMAHA], RouteCost(TrainColor.WILD, 2), 9),  # not a typo, there's two
            11: Route(11, [City.KANSAS_CITY, City.OKLAHOMA_CITY], RouteCost(TrainColor.WILD, 2), 12),
            12: Route(12, [City.KANSAS_CITY, City.OKLAHOMA_CITY], RouteCost(TrainColor.WILD, 2), 11), # not a typo, there's two
            13: Route(13, [City.DALLAS, City.OKLAHOMA_CITY], RouteCost(TrainColor.WILD, 2), 14),
            14: Route(14, [City.DALLAS, City.OKLAHOMA_CITY], RouteCost(TrainColor.WILD, 2), 13),  # not a typo, there's two
            15: Route(15, [City.ATLANTA, City.RALEIGH], RouteCost(TrainColor.WILD, 2), 16),
            16: Route(16, [City.ATLANTA, City.RALEIGH], RouteCost(TrainColor.WILD, 2), 15),  # not a typo, there's two
            17: Route(17, [City.WASHINGTON, City.RALEIGH], RouteCost(TrainColor.WILD, 2), 18),
            18: Route(18, [City.WASHINGTON, City.RALEIGH], RouteCost(TrainColor.WILD, 2), 17),  # not a typo, there's two
            19: Route(19, [City.BOSTON, City.MONTREAL], RouteCost(TrainColor.WILD, 2), 20),
            20: Route(20, [City.BOSTON, City.MONTREAL], RouteCost(TrainColor.WILD, 2), 19),  # not a typo, there's two
            21: Route(21, [City.OKLAHOMA_CITY, City.LITTLE_ROCK], RouteCost(TrainColor.WILD, 2)),
            22: Route(22, [City.DALLAS, City.LITTLE_ROCK], RouteCost(TrainColor.WILD, 2)),
            23: Route(23, [City.SAINT_LOUIS, City.LITTLE_ROCK], RouteCost(TrainColor.WILD, 2)),
            24: Route(24, [City.SAINT_LOUIS, City.NASHVILLE], RouteCost(TrainColor.WILD, 2)),
            25: Route(25, [City.RALEIGH, City.PITTSBURGH], RouteCost(TrainColor.WILD, 2)),
            26: Route(26, [City.PITTSBURGH, City.WASHINGTON], RouteCost(TrainColor.WILD, 2)),
            27: Route(27, [City.PITTSBURGH, City.TORONTO], RouteCost(TrainColor.WILD, 2)),
            28: Route(28, [City.HOUSTON, City.NEW_ORLEANS], RouteCost(TrainColor.WILD, 2)),
            29: Route(29, [City.ATLANTA, City.CHARLESTON], RouteCost(TrainColor.WILD, 2)),
            30: Route(30, [City.RALEIGH, City.CHARLESTON], RouteCost(TrainColor.WILD, 2)),
            31: Route(31, [City.LOS_ANGELES, City.LAS_VEGAS], RouteCost(TrainColor.WILD, 2)),
            32: Route(32, [City.SANTA_FE, City.DENVER], RouteCost(TrainColor.WILD, 2)),
            33: Route(33, [City.SANTA_FE, City.EL_PASO], RouteCost(TrainColor.WILD, 2)),
            34: Route(34, [City.SAULT_ST_MARIE, City.TORONTO], RouteCost(TrainColor.WILD, 2)),
            35: Route(35, [City.MONTREAL, City.TORONTO], RouteCost(TrainColor.WILD, 3)),
            36: Route(36, [City.PHOENIX, City.EL_PASO], RouteCost(TrainColor.WILD, 3)),
            37: Route(37, [City.PHOENIX, City.SANTA_FE], RouteCost(TrainColor.WILD, 3)),
            38: Route(38, [City.PHOENIX, City.LOS_ANGELES], RouteCost(TrainColor.WILD, 3)),
            39: Route(39, [City.VANCOUVER, City.CALGARY], RouteCost(TrainColor.WILD, 3)),
            40: Route(40, [City.DULUTH, City.SAULT_ST_MARIE], RouteCost(TrainColor.WILD, 3)),
            41: Route(41, [City.SEATTLE, City.CALGARY], RouteCost(TrainColor.WILD, 4)),
            42: Route(42, [City.CALGARY, City.HELENA], RouteCost(TrainColor.WILD, 4)),

            # GREENS
            43: Route(43, [City.SAINT_LOUIS, City.CHICAGO], RouteCost(TrainColor.GREEN, 2), 79),
            44: Route(44, [City.PITTSBURGH, City.NEW_YORK], RouteCost(TrainColor.GREEN, 2), 78),
            45: Route(45, [City.LITTLE_ROCK, City.NEW_ORLEANS], RouteCost(TrainColor.GREEN, 3)),
            46: Route(46, [City.HELENA, City.DENVER], RouteCost(TrainColor.GREEN, 4)),
            47: Route(47, [City.SAINT_LOUIS, City.PITTSBURGH], RouteCost(TrainColor.GREEN, 5)),
            48: Route(48, [City.PORTLAND, City.SAN_FRANCISCO], RouteCost(TrainColor.GREEN, 5), 90),
            49: Route(49, [City.EL_PASO, City.HOUSTON], RouteCost(TrainColor.GREEN, 6)),

            # BLUES
            50: Route(50, [City.KANSAS_CITY, City.SAINT_LOUIS], RouteCost(TrainColor.BLUE, 2), 85),
            51: Route(51, [City.SANTA_FE, City.OKLAHOMA_CITY], RouteCost(TrainColor.BLUE, 3)),
            52: Route(52, [City.NEW_YORK, City.MONTREAL], RouteCost(TrainColor.BLUE, 3)),
            53: Route(53, [City.OMAHA, City.CHICAGO], RouteCost(TrainColor.BLUE, 4)),
            54: Route(54, [City.HELENA, City.WINNIPEG], RouteCost(TrainColor.BLUE, 4)),
            55: Route(55, [City.ATLANTA, City.MIAMI], RouteCost(TrainColor.BLUE, 5)),
            56: Route(56, [City.PORTLAND, City.SALT_LAKE_CITY], RouteCost(TrainColor.BLUE, 6)),

            # REDS
            57: Route(57, [City.NEW_YORK, City.BOSTON], RouteCost(TrainColor.RED, 2), 92),
            58: Route(58, [City.DULUTH, City.CHICAGO], RouteCost(TrainColor.RED, 3)),
            59: Route(59, [City.SALT_LAKE_CITY, City.DENVER], RouteCost(TrainColor.RED, 3), 93),
            60: Route(60, [City.DENVER, City.OKLAHOMA_CITY], RouteCost(TrainColor.RED, 4)),
            61: Route(61, [City.EL_PASO, City.DALLAS], RouteCost(TrainColor.RED, 4)),
            62: Route(62, [City.HELENA, City.OMAHA], RouteCost(TrainColor.RED, 5)),
            63: Route(63, [City.NEW_ORLEANS, City.MIAMI], RouteCost(TrainColor.RED, 6)),

            # ORANGES
            64: Route(64, [City.WASHINGTON, City.NEW_YORK], RouteCost(TrainColor.ORANGE, 2), 71),
            65: Route(65, [City.CHICAGO, City.PITTSBURGH], RouteCost(TrainColor.ORANGE, 3), 73),
            66: Route(66, [City.LAS_VEGAS, City.SALT_LAKE_CITY], RouteCost(TrainColor.ORANGE, 3)),
            67: Route(67, [City.NEW_ORLEANS, City.ATLANTA], RouteCost(TrainColor.ORANGE, 4), 96),
            68: Route(68, [City.DENVER, City.KANSAS_CITY], RouteCost(TrainColor.ORANGE, 4), 74),
            69: Route(69, [City.SAN_FRANCISCO, City.SALT_LAKE_CITY], RouteCost(TrainColor.ORANGE, 5), 83),
            70: Route(70, [City.HELENA, City.DULUTH], RouteCost(TrainColor.ORANGE, 6)),

            # BLACKS
            71: Route(71, [City.WASHINGTON, City.NEW_YORK], RouteCost(TrainColor.BLACK, 2), 64),
            72: Route(72, [City.NASHVILLE, City.RALEIGH], RouteCost(TrainColor.BLACK, 3)),
            73: Route(73, [City.CHICAGO, City.PITTSBURGH], RouteCost(TrainColor.BLACK, 3), 65),
            74: Route(74, [City.DENVER, City.KANSAS_CITY], RouteCost(TrainColor.BLACK, 4), 68),
            75: Route(75, [City.WINNIPEG, City.DULUTH], RouteCost(TrainColor.BLACK, 4)),
            76: Route(76, [City.SAULT_ST_MARIE, City.MONTREAL], RouteCost(TrainColor.BLACK, 5)),
            77: Route(77, [City.LOS_ANGELES, City.EL_PASO], RouteCost(TrainColor.BLACK, 6)),

            # WHITES
            78: Route(78, [City.PITTSBURGH, City.NEW_YORK], RouteCost(TrainColor.WHITE, 2), 44),
            79: Route(79, [City.SAINT_LOUIS, City.CHICAGO], RouteCost(TrainColor.WHITE, 2), 43),
            80: Route(80, [City.LITTLE_ROCK, City.NASHVILLE], RouteCost(TrainColor.WHITE, 3)),
            81: Route(81, [City.CHICAGO, City.TORONTO], RouteCost(TrainColor.WHITE, 4)),
            82: Route(82, [City.DENVER, City.PHOENIX], RouteCost(TrainColor.WHITE, 5)),
            83: Route(83, [City.SAN_FRANCISCO, City.SALT_LAKE_CITY], RouteCost(TrainColor.WHITE, 5), 69),
            84: Route(84, [City.CALGARY, City.WINNIPEG], RouteCost(TrainColor.WHITE, 6)),

            # PINKS
            85: Route(85, [City.KANSAS_CITY, City.SAINT_LOUIS], RouteCost(TrainColor.PINK, 2), 50),
            86: Route(86, [City.LOS_ANGELES, City.SAN_FRANCISCO], RouteCost(TrainColor.PINK, 3), 94),
            87: Route(87, [City.SALT_LAKE_CITY, City.HELENA], RouteCost(TrainColor.PINK, 3)),
            88: Route(88, [City.DENVER, City.OMAHA], RouteCost(TrainColor.PINK, 4)),
            89: Route(89, [City.CHARLESTON, City.MIAMI], RouteCost(TrainColor.PINK, 4)),
            90: Route(90, [City.SAN_FRANCISCO, City.PORTLAND], RouteCost(TrainColor.PINK, 5), 48),
            91: Route(91, [City.DULUTH, City.TORONTO], RouteCost(TrainColor.PINK, 6)),

            # YELLOWS
            92: Route(92, [City.NEW_YORK, City.BOSTON], RouteCost(TrainColor.YELLOW, 2), 57),
            93: Route(93, [City.SALT_LAKE_CITY, City.DENVER], RouteCost(TrainColor.YELLOW, 3), 59),
            94: Route(94, [City.LOS_ANGELES, City.SAN_FRANCISCO], RouteCost(TrainColor.YELLOW, 3), 86),
            95: Route(95, [City.NASHVILLE, City.PITTSBURGH], RouteCost(TrainColor.YELLOW, 4)),
            96: Route(96, [City.NEW_ORLEANS, City.ATLANTA], RouteCost(TrainColor.YELLOW, 4), 68),
            97: Route(97, [City.EL_PASO, City.OKLAHOMA_CITY], RouteCost(TrainColor.YELLOW, 5)),
            98: Route(98, [City.SEATTLE, City.HELENA], RouteCost(TrainColor.YELLOW, 6)),
        }

        self.destinations: Dict[Destination] = {
            # ([city1, city2], points)
            0: Destination([City.BOSTON, City.MIAMI], 12),
            1: Destination([City.CALGARY, City.PHOENIX], 13),
            2: Destination([City.CALGARY, City.SALT_LAKE_CITY], 7),
            3: Destination([City.CHICAGO, City.NEW_ORLEANS], 7),
            4: Destination([City.CHICAGO, City.SANTA_FE], 9),
            5: Destination([City.DALLAS, City.NEW_YORK], 11),
            6: Destination([City.DENVER, City.EL_PASO], 4),
            7: Destination([City.DENVER, City.PITTSBURGH], 11),
            8: Destination([City.DULUTH, City.EL_PASO], 10),
            9: Destination([City.DULUTH, City.HOUSTON], 8),
            10: Destination([City.HELENA, City.LOS_ANGELES], 8),
            11: Destination([City.KANSAS_CITY, City.HOUSTON], 5),
            12: Destination([City.LOS_ANGELES, City.CHICAGO], 16),
            13: Destination([City.LOS_ANGELES, City.MIAMI], 20),
            14: Destination([City.LOS_ANGELES, City.NEW_YORK], 21),
            15: Destination([City.MONTREAL, City.ATLANTA], 9),
            16: Destination([City.MONTREAL, City.NEW_ORLEANS], 13),
            17: Destination([City.NEW_YORK, City.ATLANTA], 6),
            18: Destination([City.PORTLAND, City.NASHVILLE], 17),
            19: Destination([City.PORTLAND, City.PHOENIX], 11),
            20: Destination([City.SAN_FRANCISCO, City.ATLANTA], 17),
            21: Destination([City.SAULT_ST_MARIE, City.NASHVILLE], 8),
            22: Destination([City.SAULT_ST_MARIE, City.OKLAHOMA_CITY], 9),
            23: Destination([City.SEATTLE, City.LOS_ANGELES], 9),
            24: Destination([City.SEATTLE, City.NEW_YORK], 22),
            25: Destination([City.TORONTO, City.MIAMI], 10),
            26: Destination([City.VANCOUVER, City.MONTREAL], 20),
            27: Destination([City.VANCOUVER, City.SANTA_FE], 13),
            28: Destination([City.WINNIPEG, City.HOUSTON], 12),
            29: Destination([City.WINNIPEG, City.LITTLE_ROCK], 11)
        }
