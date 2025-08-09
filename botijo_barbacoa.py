# -*- coding: utf-8 -*-
import random, threading, time
from typing import Callable, Dict, List, Optional
from phrases_botijo import PHRASES, ALL_150

class PhraseManager:
    def __init__(self, seed: Optional[int] = None):
        self.random = random.Random(seed)
        self.pool_all = list(ALL_150)
        self.pool_by_cat: Dict[str, List[str]] = {k: list(v) for k, v in PHRASES.items()}
        self.used_all: List[str] = []
        self.used_by_cat: Dict[str, List[str]] = {k: [] for k in PHRASES}

    def _pop_from(self, arr: List[str]) -> str:
        if not arr:
            arr.extend(self.used_all)
            self.used_all.clear()
        choice = self.random.choice(arr)
        arr.remove(choice)
        self.used_all.append(choice)
        return choice

    def any(self) -> str:
        return self._pop_from(self.pool_all)

    def from_cat(self, cat: str) -> str:
        if cat not in self.pool_by_cat:
            return self.any()
        pool = self.pool_by_cat[cat]
        if not pool:
            pool.extend(self.used_by_cat[cat])
            self.used_by_cat[cat].clear()
        choice = random.choice(pool)
        pool.remove(choice)
        self.used_by_cat[cat].append(choice)
        return choice

class AutoBanterThread(threading.Thread):
    def __init__(
        self,
        speaker: Callable[[str], None],
        phrase_source: PhraseManager,
        min_s: int = 120,
        max_s: int = 240,
        consult_is_speaking: Optional[Callable[[], bool]] = None,
        preferred_cats: Optional[List[str]] = None,
    ):
        super().__init__(daemon=True)
        self.speaker = speaker
        self.src = phrase_source
        self.min_s = min_s
        self.max_s = max_s
        self.consult_is_speaking = consult_is_speaking or (lambda: False)
        self.stop_event = threading.Event()
        self.preferred_cats = preferred_cats or []

    def stop(self): self.stop_event.set()

    def run(self):
        while not self.stop_event.is_set():
            wait = random.randint(self.min_s, self.max_s)
            for _ in range(wait * 10):
                if self.stop_event.is_set(): return
                time.sleep(0.1)
            if self.consult_is_speaking(): 
                continue
            try:
                if self.preferred_cats:
                    cat = random.choice(self.preferred_cats)
                    text = self.src.from_cat(cat)
                else:
                    text = self.src.any()
                self.speaker(text)
            except Exception as e:
                print(f"[AUTOBANTER] {e}")

class FaceBanter:
    def __init__(
        self,
        speaker: Callable[[str], None],
        phrase_source: PhraseManager,
        cooldown_s: int = 35,
        category_sequence: Optional[List[str]] = None,
        consult_is_speaking: Optional[Callable[[], bool]] = None,
    ):
        self.speaker = speaker
        self.src = phrase_source
        self.cooldown_s = cooldown_s
        self.last_time = 0.0
        self.consult_is_speaking = consult_is_speaking or (lambda: False)
        self.category_sequence = category_sequence or ["ninos_humor_blanco","fuego_barbacoa","aristocracia_culta","interaccion_directa"]

    def maybe_emit(self):
        if self.consult_is_speaking(): 
            return
        now = time.time()
        if now - self.last_time < self.cooldown_s:
            return
        cat = self.category_sequence[int(now) % len(self.category_sequence)]
        self.last_time = now
        try:
            self.speaker(self.src.from_cat(cat))
        except Exception as e:
            print(f"[FACEBANTER] {e}")

def make_setup(
    speaker: Callable[[str], None],
    consult_is_speaking: Optional[Callable[(), bool]] = None,
    periodic_min_s: int = 120,
    periodic_max_s: int = 240,
    face_cooldown_s: int = 35,
) -> dict:
    pm = PhraseManager()
    auto = AutoBanterThread(
        speaker=speaker,
        phrase_source=pm,
        min_s=periodic_min_s,
        max_s=periodic_max_s,
        consult_is_speaking=consult_is_speaking,
        preferred_cats=["fuego_barbacoa","carne_comida","aristocracia_culta"],
    )
    face = FaceBanter(
        speaker=speaker,
        phrase_source=pm,
        cooldown_s=face_cooldown_s,
        category_sequence=["ninos_humor_blanco","fuego_barbacoa","aristocracia_culta","interaccion_directa"],
        consult_is_speaking=consult_is_speaking,
    )
    return {"phrases": pm, "auto": auto, "face": face}
