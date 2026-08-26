"""One published NYT Friday, for the real-clue evaluation stratum.

Friday, May 28, 2021 by Andrew J. Ries, edited by Will Shortz. Copyright the
New York Times Company; included here only so the agent can be scored on
contemporary clues. The filled ``.xd`` is written to ``corpus/nyt/`` (gitignored)
and is not redistributed with the public-domain generated corpus.
"""

from __future__ import annotations

import os

from ..model import Puzzle
from ..xd import dump_xd, parse_xd

GRID = """\
BEERBAR##BATHED
ANGELPIE#AREOLA
LOOSEEND#LEANER
ESSIE#DIGS#LONE
###GDP#TEASERAD
TRANSICON#EAR##
HERS#TORTREFORM
AFT#ACMILAN#LIE
DICKCHENEY#ALPO
##RAE#SCROLLSAW
WEIRDHUH#NAB###
ALTA#EPIC#TESTS
TAICHI#EASTROOM
ETCHES#FREETIME
RESIST##SARALEE
"""

ACROSS = {
    1: "Building with many drafts",
    8: "Got clean",
    14: '"Heavenly" dessert with a lemony filling',
    16: "Space between the ribs of an insect wing",
    17: "Unresolved detail",
    18: "One-pointer in horseshoes",
    19: "Woman's name that sounds like two letters of the alphabet",
    20: "Pad",
    22: "Without a match",
    23: "Econ. stat",
    25: "Enticing spot",
    27: "Gay rights pioneer Marsha P. Johnson, for one",
    31: "Bud's place",
    32: '"I\'m ___, she\'s mine" (repeated lyric in "Do Wah Diddy Diddy")',
    33: "Movement to reduce frivolous lawsuits",
    37: "Early p.m.",
    38: "Renowned football club founded in 1899",
    39: "Golfer's concern",
    40: '"Vice" principal?',
    42: "T-Bonz treats brand",
    43: "Bob ___, Canadian ambassador to the U.N.",
    44: "Curve cutter",
    46: '"Isn\'t that strange?"',
    49: "Collar",
    50: 'Resort with a "no snowboarders" policy',
    51: "More than outstanding",
    53: "They produce results",
    57: "Discipline based on the principles of yin and yang",
    59: "White House reception locale",
    61: "Leaves a grave impression, perhaps",
    62: "What's not working?",
    63: "Protester's cry",
    64: "Brand whose famous slogan contains a double negative",
}

DOWN = {
    1: "Roll in the hay?",
    2: "905-year-old in Genesis",
    3: "Clash of the titans?",
    4: "Throws in the towel",
    5: "Runs",
    6: "Great ___",
    7: "Tough skin",
    8: "Model's makeup, often",
    9: "Is for more?",
    10: "Bit for a fortuneteller",
    11: "90s groups?",
    12: "First name on the Supreme Court",
    13: "Had the gall",
    15: "Post master?",
    21: "Not so hard",
    24: "Topic for a voice coach",
    26: "Taken in",
    27: "Jazz trumpeter Jones",
    28: "Offer you might have less interest in, for short?",
    29: "Ones who might use oils in a pan?",
    30: "Surfaces",
    34: "Fabric made from cellulose",
    35: "Seacrest's partner on morning TV",
    36: "Cat's 'sup?",
    38: "Didn't just excel on",
    41: "Home of Jinnah International Airport",
    42: "Province named for a daughter of Queen Victoria",
    45: "Second",
    46: "Word with hold or holy",
    47: "Lift up",
    48: "Job that usually requires a face covering",
    52: "Parts of roller coasters",
    54: "Potter's substance",
    55: "Antithesis of light reading?",
    56: "Aide to Captain Hook",
    58: '"___ Your Dog, Charlie Brown" (TV special)',
    60: '"Finding Nemo" setting',
}


def xd_source() -> str:
    clues = []
    for number, clue in sorted(ACROSS.items()):
        clues.append(f"A{number}. {clue}")
    clues.append("")
    for number, clue in sorted(DOWN.items()):
        clues.append(f"D{number}. {clue}")
    return (
        "Title: New York Times, Friday, May 28, 2021\n"
        "Author: Andrew J. Ries\n"
        "Editor: Will Shortz\n"
        "Copyright: 2021 The New York Times Company\n"
        "Date: 2021-05-28\n"
        "Source: nyt\n"
        "id: nyt-2021-05-28\n"
        "\n\n"
        + GRID.strip()
        + "\n\n\n"
        + "\n".join(clues)
        + "\n"
    )


def build_puzzle() -> Puzzle:
    return parse_xd(xd_source(), puzzle_id="nyt-2021-05-28")


def write_corpus(directory: str | None = None) -> str:
    """Write the filled .xd under corpus/nyt/. Returns the path."""
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    directory = directory or os.path.join(here, "corpus", "nyt")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "nyt-2021-05-28.xd")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(dump_xd(build_puzzle()))
    return path
