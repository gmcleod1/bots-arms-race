"""Organic-looking text for synthetic humans.

Posts come from a slot grammar (ten templates over adjectives, feelings, times,
verbs and two community-specific nouns), which gives tens of thousands of
distinct sentences per community. Each person then writes in a personal style:
habitual misspellings that recur, a random typo rate, casing, punctuation and
emoji habits.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

# One noun list per community (16 each). Length must match world.N_COMMUNITIES.
NOUNS = [
    ["controller", "raid", "speedrun", "console", "boss", "loadout", "lobby", "mod",
     "stream", "patch", "quest", "headset", "ranking", "save", "guild", "playthrough"],
    ["sourdough", "skillet", "marinade", "casserole", "risotto", "pantry", "brisket", "glaze",
     "ferment", "stockpot", "omelet", "dumplings", "chili", "tart", "broth", "spice"],
    ["deadlift", "treadmill", "marathon", "kettlebell", "sprint", "stretch", "protein", "squat",
     "cardio", "trail", "rowing", "pullup", "yoga", "pacer", "recovery", "gym"],
    ["chorus", "vinyl", "setlist", "bassline", "playlist", "encore", "riff", "synth",
     "demo", "album", "lyrics", "drummer", "concert", "remix", "melody", "mixtape"],
    ["itinerary", "hostel", "layover", "passport", "backpack", "roadtrip", "ferry", "campsite",
     "souvenir", "visa", "skyline", "airport", "hike", "beach", "railway", "detour"],
    ["puppy", "kitten", "leash", "aquarium", "hamster", "collar", "shelter", "parrot",
     "vet", "treat", "litter", "groomer", "terrier", "tabby", "kennel", "fetch"],
    ["laptop", "keyboard", "firmware", "router", "monitor", "script", "server", "battery",
     "browser", "kernel", "gadget", "charger", "motherboard", "cable", "sensor", "widget"],
    ["sequel", "trailer", "screenplay", "premiere", "soundtrack", "director", "cliffhanger", "finale",
     "reboot", "cameo", "spoiler", "credits", "rerun", "subtitle", "cinema", "franchise"],
]
ADJECTIVES = [
    "new", "old", "broken", "shiny", "weird", "cheap", "huge", "tiny", "loud", "quiet", "messy",
    "perfect", "awful", "lucky", "slow", "fast", "fancy", "rusty", "cozy", "wild", "fresh",
    "stubborn", "random", "solid", "gorgeous", "clumsy", "ancient", "bright", "dusty", "sturdy",
    "flimsy", "noisy", "crisp", "bitter", "smooth", "heavy", "glossy", "sketchy", "clever",
    "dented", "sleek", "faded", "bulky", "polished", "tangled", "vintage", "modest", "chaotic",
]
FEELINGS = [
    "happy", "annoyed", "nervous", "excited", "tired", "confused", "proud", "sad", "hyped",
    "relieved", "bored", "anxious", "thrilled", "grumpy", "amazed", "calm", "jealous",
    "embarrassed", "hopeful", "over it", "curious", "restless", "grateful", "pumped", "numb",
    "giddy", "stressed", "content", "dizzy", "tense", "sleepy", "smug", "shaky", "uneasy",
    "cheerful", "fed up", "overwhelmed", "lonely", "determined", "sore",
]
TIMES = [
    "this morning", "last night", "today", "yesterday", "all weekend", "after work", "at lunch",
    "on monday", "this week", "just now", "earlier", "on friday", "this afternoon",
    "around midnight", "over the holidays", "on tuesday", "last month", "a while ago",
    "before breakfast", "during the commute", "on saturday", "two days ago", "this evening",
    "right before bed",
]
VERBS = [
    "fix", "replace", "rebuild", "return", "test", "clean", "upgrade", "finish", "start",
    "redo", "sell", "borrow", "learn", "share", "skip", "try", "assemble", "polish", "organize",
    "unpack", "rewire", "measure", "bake", "train", "film", "record", "repair", "order", "pack",
    "label", "tune", "compare",
]
# A post is one opening clause, a connector, and one closing clause: over a thousand
# skeletons, each with several free slots, so unrelated humans rarely write near-copies.
CLAUSES_A = [
    "just finished my {adj} {noun}",
    "{time} I was really thinking about my {noun}",
    "not gonna lie the {noun} {time} was {feel}",
    "definitely need a new {noun}, mine is {adj}",
    "spent {time} with the {adj} {noun}",
    "finally got the {noun} working",
    "why is the {noun} always {adj}",
    "anyone else going to try the {adj} {noun} tomorrow?",
    "{time} update: the {adj} {noun} is {feel}",
    "hot take: the {noun} is really {adj}",
    "still thinking about that {adj} {noun} from {time}",
    "the {noun} is {feel} and honestly so am I",
]
CLAUSES_B = [
    "the {noun2} is {adj2} compared with the {noun3}",
    "I am just going to {verb} it with my {noun2}",
    "because of the {noun3} I am feeling {feel2}",
    "my {noun2} and my {noun3} are both {adj2}",
    "really need to {verb} the {adj2} {noun2} soon",
    "the {noun3} made me {feel2} about the {noun2}",
    "maybe I should {verb} the {noun3} with a {adj2} {noun2}",
    "honestly the {noun2} is the {adj2} part",
    "with the {noun3} and the {noun2} I am {feel2}",
    "going to {verb} my {noun3} before the {noun2} gets {adj2}",
    "just hoping the {noun2} stays {adj2} this time",
    "definitely not going to {verb} the {noun3} again",
]
CONNECTORS = [", and", ", but", " so", " because", ";", ". Also", ", then", " and"]
POST_TEMPLATES = CLAUSES_A + CLAUSES_B  # every template a post can be built from
COMMENT_TEMPLATES = [
    "{react}",
    "{react} {noun}",
    "same, my {noun} is {adj}",
    "{react} where did you get the {noun}?",
    "{react}, so {feel}",
]
REACTS = [
    "lol", "same", "this", "love this", "no way", "so true", "facts", "big mood", "wow",
    "haha", "ok wow", "nice", "yikes", "respect", "mood",
]

# (correct, misspelling): habits some people have. Frequent words so habits show up.
HABIT_TYPOS = (
    ("the", "teh"), ("because", "becuase"), ("really", "realy"), ("definitely", "definately"),
    ("tomorrow", "tomorow"), ("with", "wiht"), ("just", "jsut"), ("going", "goign"),
)
END_PUNCT = ("", "", ".", "!", "!!", "...")
EMOJI = ("\U0001F602", "\U0001F525", "\U0001F62D", "\U0001F44D", "\U0001F60A", "\U0001F914", "\U0001F480")


@dataclass(frozen=True)
class WritingStyle:
    typo_rate: float  # chance a given word (4+ letters) gets a random typo
    habits: tuple[tuple[str, str], ...]  # habitual misspellings, applied most of the time
    lowercase: bool  # never capitalises
    end_punct: str
    emoji: tuple[str, ...]  # this person's favourite emoji
    emoji_rate: float


def make_style(rng: np.random.Generator) -> WritingStyle:
    n_habits = int(rng.choice((0, 1, 2), p=(0.2, 0.5, 0.3)))
    picks = rng.choice(len(HABIT_TYPOS), size=n_habits, replace=False)
    habits = tuple(HABIT_TYPOS[int(i)] for i in sorted(picks))
    n_emoji = int(rng.choice((0, 1, 2), p=(0.4, 0.4, 0.2)))
    emoji = tuple(EMOJI[int(i)] for i in sorted(rng.choice(len(EMOJI), size=n_emoji, replace=False)))
    return WritingStyle(
        typo_rate=float(rng.lognormal(mean=np.log(0.012), sigma=0.7)),
        habits=habits,
        lowercase=bool(rng.random() < 0.3),
        end_punct=str(rng.choice(END_PUNCT)),
        emoji=emoji,
        emoji_rate=float(rng.uniform(0.1, 0.4)) if emoji else 0.0,
    )


def _pick(rng: np.random.Generator, options):
    return options[int(rng.integers(len(options)))]


def _slots(rng: np.random.Generator, community: int) -> dict[str, str]:
    """Always draws every slot so RNG consumption is constant per call."""
    nouns = NOUNS[community]
    n1, n2, n3 = (int(i) for i in rng.choice(len(nouns), size=3, replace=False))  # three distinct
    return {
        "noun": nouns[n1], "noun2": nouns[n2], "noun3": nouns[n3],
        "adj": _pick(rng, ADJECTIVES), "adj2": _pick(rng, ADJECTIVES),
        "feel": _pick(rng, FEELINGS), "feel2": _pick(rng, FEELINGS),
        "time": _pick(rng, TIMES), "verb": _pick(rng, VERBS), "react": _pick(rng, REACTS),
    }


def _random_typo(rng: np.random.Generator, word: str) -> str:
    i = int(rng.integers(1, len(word) - 1))
    kind = int(rng.integers(3))
    if kind == 0:  # swap two adjacent letters
        return word[:i] + word[i + 1] + word[i] + word[i + 2 :]
    if kind == 1:  # drop a letter
        return word[:i] + word[i + 1 :]
    return word[:i] + word[i] + word[i:]  # double a letter


def apply_style(rng: np.random.Generator, text: str, style: WritingStyle) -> str:
    habit = dict(style.habits)
    out = []
    for word in text.split(" "):
        core = word.strip(".,!?:;").lower()
        if core in habit and rng.random() < 0.6:
            word = word.lower().replace(core, habit[core])
        elif len(core) >= 4 and rng.random() < style.typo_rate:
            word = word.lower().replace(core, _random_typo(rng, core))
        out.append(word)
    text = " ".join(out)
    text = text.lower() if style.lowercase else text[:1].upper() + text[1:]
    text = text.rstrip(".!?") + style.end_punct
    if style.emoji and rng.random() < style.emoji_rate:
        text += " " + _pick(rng, style.emoji)
    return text


def compose(rng: np.random.Generator, action: str, community: int, style: WritingStyle) -> str:
    if action == "post":
        template = _pick(rng, CLAUSES_A) + _pick(rng, CONNECTORS) + " " + _pick(rng, CLAUSES_B)
    else:
        template = _pick(rng, COMMENT_TEMPLATES)
    return apply_style(rng, template.format(**_slots(rng, community)), style)


@lru_cache(maxsize=1)
def vocabulary() -> frozenset[str]:
    """Every correctly spelled word the generator can emit (lowercase letters only)."""
    words: set[str] = set()
    sources = list(POST_TEMPLATES) + list(COMMENT_TEMPLATES) + CONNECTORS + REACTS
    sources += ADJECTIVES + FEELINGS + TIMES + VERBS + [n for group in NOUNS for n in group]
    for s in sources:
        words.update(re.findall(r"[a-z]+", re.sub(r"\{[a-z0-9]+\}", " ", s.lower())))
    return frozenset(words)
