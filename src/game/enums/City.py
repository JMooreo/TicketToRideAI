from enum import Enum


class City(Enum):
    def __str__(self):
        return self.name

    VANCOUVER = 1
    CALGARY = 2
    WINNIPEG = 3
    SAULT_ST_MARIE = 4
    MONTREAL = 5
    SEATTLE = 6
    DULUTH = 7
    TORONTO = 8
    BOSTON = 9
    PORTLAND = 10
    HELENA = 11
    OMAHA = 12
    CHICAGO = 13
    PITTSBURGH = 14
    SAN_FRANCISCO = 15
    SALT_LAKE_CITY = 16
    DENVER = 17
    KANSAS_CITY = 18
    SAINT_LOUIS = 19
    NASHVILLE = 20
    WASHINGTON = 21
    RALEIGH = 22
    ATLANTA = 23
    CHARLESTON = 24
    LOS_ANGELES = 25
    PHOENIX = 26
    SANTA_FE = 27
    DALLAS = 28
    EL_PASO = 29
    HOUSTON = 30
    NEW_ORLEANS = 31
    MIAMI = 32
    NEW_YORK = 33
    OKLAHOMA_CITY = 34
    LITTLE_ROCK = 35
    LAS_VEGAS = 36
