"""
Improved Ollama LLM as Optimizer for Spotify playlist optimization.

Run example:
    python main.py --dataset dataset.csv --model qwen2.5:0.5b --iterations 20 --initial random

Before running:
    ollama pull qwen2.5:0.5b
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


TARGET_PROFILES = {
    "party": {
        "danceability": 0.82,
        "energy": 0.78,
        "valence": 0.72,
        "tempo": 122,
        "acousticness": 0.25,
        "instrumentalness": 0.08,
    },
    "study": {
        "danceability": 0.45,
        "energy": 0.35,
        "valence": 0.50,
        "tempo": 95,
        "acousticness": 0.55,
        "instrumentalness": 0.35,
    },
    "workout": {
        "danceability": 0.75,
        "energy": 0.88,
        "valence": 0.70,
        "tempo": 135,
        "acousticness": 0.15,
        "instrumentalness": 0.05,
    },
    "chill": {
        "danceability": 0.55,
        "energy": 0.35,
        "valence": 0.55,
        "tempo": 90,
        "acousticness": 0.60,
        "instrumentalness": 0.20,
    },
}

SCORE_WEIGHTS = {
    "mood_fit": 0.35,
    "smoothness": 0.25,
    "diversity": 0.20,
    "popularity": 0.10,
    "duration_fit": 0.10,
}


class PlaylistProblem:
    def __init__(
        self,
        dataset_path: str,
        scenario: str = "party",
        playlist_size: int = 20,
        candidate_pool_size: int = 800,
        target_duration_min: float = 60.0,
        seed: int = 42,
    ):
        self.dataset_path = Path(dataset_path)
        self.scenario = scenario
        self.playlist_size = playlist_size
        self.candidate_pool_size = candidate_pool_size
        self.target_duration_min = target_duration_min
        self.seed = seed
        self.target_profile = TARGET_PROFILES[scenario]

        random.seed(seed)
        np.random.seed(seed)

        self.data = self.load_data()
        self.candidate_pool = self.build_candidate_pool()

    def load_data(self) -> pd.DataFrame:
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {self.dataset_path}. "
                "Put dataset.csv into the project folder or pass --dataset path/to/dataset.csv"
            )

        df = pd.read_csv(self.dataset_path)
        df.columns = [str(c).strip().lower() for c in df.columns]

        if "track_genre" in df.columns and "genre" not in df.columns:
            df = df.rename(columns={"track_genre": "genre"})

        required = [
            "track_name", "artists", "popularity", "duration_ms",
            "danceability", "energy", "valence", "tempo",
            "acousticness", "instrumentalness", "genre"
        ]

        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in dataset: {missing}")

        data = df[required].copy()
        data = data.dropna(subset=required)
        data = data.drop_duplicates(subset=["track_name", "artists"]).reset_index(drop=True)

        numeric = [
            "popularity", "duration_ms", "danceability", "energy",
            "valence", "tempo", "acousticness", "instrumentalness"
        ]

        for col in numeric:
            data[col] = pd.to_numeric(data[col], errors="coerce")

        data = data.dropna(subset=numeric)
        data = data[(data["tempo"] >= 50) & (data["tempo"] <= 220)].reset_index(drop=True)
        data["duration_min"] = data["duration_ms"] / 60000
        data["individual_mood_fit"] = data.apply(self.individual_mood_fit, axis=1)
        return data

    def build_candidate_pool(self) -> pd.DataFrame:
        # A larger pool makes the random initial playlist weaker and leaves room for improvement.
        pool = (
            self.data.sort_values(["individual_mood_fit", "popularity"], ascending=False)
            .head(self.candidate_pool_size)
            .copy()
            .reset_index(drop=True)
        )
        pool["cid"] = pool.index
        return pool

    def norm_abs_diff(self, value: float, target: float, feature: str) -> float:
        denom = 170 if feature == "tempo" else 1
        return min(abs(float(value) - float(target)) / denom, 1.0)

    def individual_mood_fit(self, row: pd.Series) -> float:
        weights = {
            "danceability": 0.25,
            "energy": 0.25,
            "valence": 0.20,
            "tempo": 0.15,
            "acousticness": 0.10,
            "instrumentalness": 0.05,
        }

        score = 0.0
        for feature, weight in weights.items():
            distance = self.norm_abs_diff(row[feature], self.target_profile[feature], feature)
            score += weight * (1 - distance)

        return float(score)

    def transition_score(self, a: pd.Series, b: pd.Series) -> float:
        diffs = {
            "tempo": abs(a["tempo"] - b["tempo"]) / 170,
            "energy": abs(a["energy"] - b["energy"]),
            "valence": abs(a["valence"] - b["valence"]),
            "danceability": abs(a["danceability"] - b["danceability"]),
        }
        weights = {
            "tempo": 0.35,
            "energy": 0.30,
            "valence": 0.20,
            "danceability": 0.15,
        }
        distance = sum(weights[k] * min(v, 1.0) for k, v in diffs.items())
        return float(max(0, 1 - distance))

    @staticmethod
    def normalized_entropy(values: list[Any]) -> float:
        if len(values) <= 1:
            return 0.0
        counts = pd.Series(values).value_counts(normalize=True)
        entropy = -sum(p * math.log(p + 1e-12) for p in counts)
        max_entropy = math.log(len(counts)) if len(counts) > 1 else 1
        return float(entropy / max_entropy) if max_entropy > 0 else 0.0

    def score_playlist(self, cids: list[int], detailed: bool = False) -> Any:
        cids = list(map(int, cids))

        unique_ratio = len(set(cids)) / max(len(cids), 1)
        length_penalty = 1 - min(abs(len(cids) - self.playlist_size) / self.playlist_size, 1)

        valid_cids = [cid for cid in cids if 0 <= cid < len(self.candidate_pool)]
        if len(valid_cids) != len(cids):
            result = {
                "score": 0.0, "mood_fit": 0.0, "smoothness": 0.0,
                "diversity": 0.0, "popularity": 0.0, "duration_fit": 0.0,
                "total_duration_min": 0.0, "unique_ratio": unique_ratio
            }
            return result if detailed else 0.0

        pl = self.candidate_pool.iloc[cids].copy()

        mood_fit = float(pl["individual_mood_fit"].mean())

        if len(pl) > 1:
            smoothness = float(np.mean([
                self.transition_score(pl.iloc[i], pl.iloc[i + 1])
                for i in range(len(pl) - 1)
            ]))
        else:
            smoothness = 0.0

        genre_diversity = self.normalized_entropy(pl["genre"].astype(str).tolist())
        artist_diversity = self.normalized_entropy(pl["artists"].astype(str).tolist())

        adjacent_same_artist = 0
        adjacent_same_genre = 0

        for i in range(len(pl) - 1):
            if str(pl.iloc[i]["artists"]) == str(pl.iloc[i + 1]["artists"]):
                adjacent_same_artist += 1
            if str(pl.iloc[i]["genre"]) == str(pl.iloc[i + 1]["genre"]):
                adjacent_same_genre += 1

        adjacent_artist_penalty = adjacent_same_artist / max(len(pl) - 1, 1)
        adjacent_genre_penalty = adjacent_same_genre / max(len(pl) - 1, 1)

        diversity = (
            0.55 * genre_diversity
            + 0.35 * artist_diversity
            + 0.10 * (1 - adjacent_genre_penalty)
        )
        diversity = max(0.0, diversity - 0.25 * adjacent_artist_penalty)
        diversity = float(np.clip(diversity, 0, 1))

        popularity = float((pl["popularity"].clip(0, 100) / 100).mean())

        total_duration_min = float(pl["duration_min"].sum())
        duration_error = abs(total_duration_min - self.target_duration_min)
        duration_fit = float(max(0, 1 - duration_error / 20))

        components = {
            "mood_fit": mood_fit,
            "smoothness": smoothness,
            "diversity": diversity,
            "popularity": popularity,
            "duration_fit": duration_fit,
        }

        score = sum(SCORE_WEIGHTS[k] * components[k] for k in SCORE_WEIGHTS)
        score *= unique_ratio
        score *= length_penalty

        result = {
            "score": float(score),
            **components,
            "total_duration_min": total_duration_min,
            "unique_ratio": unique_ratio,
        }

        return result if detailed else float(score)

    def transition_plus_mood_score(self, prev_cid: int, next_cid: int) -> float:
        a = self.candidate_pool.iloc[int(prev_cid)]
        b = self.candidate_pool.iloc[int(next_cid)]
        return 0.65 * self.transition_score(a, b) + 0.35 * float(b["individual_mood_fit"])

    def greedy_order(self, cids: list[int]) -> list[int]:
        cids = list(map(int, cids))
        if not cids:
            return []

        start = min(cids, key=lambda cid: abs(self.candidate_pool.iloc[cid]["tempo"] - self.target_profile["tempo"]))
        ordered = [start]
        remaining = set(cids)
        remaining.remove(start)

        while remaining:
            prev = ordered[-1]
            nxt = max(remaining, key=lambda cid: self.transition_plus_mood_score(prev, cid))
            ordered.append(int(nxt))
            remaining.remove(nxt)

        return ordered

    def build_initial_playlist(self, initial: str = "random") -> list[int]:
        rng = np.random.default_rng(self.seed)

        if initial == "greedy":
            top_mood = self.candidate_pool.sort_values("individual_mood_fit", ascending=False).head(120)
            chosen = rng.choice(top_mood["cid"].values, size=self.playlist_size, replace=False).tolist()
            return self.greedy_order(chosen)

        # Random weak start: choose from the whole candidate pool and keep random order.
        # This makes the convergence plot more informative.
        chosen = rng.choice(self.candidate_pool["cid"].values, size=self.playlist_size, replace=False).tolist()
        rng.shuffle(chosen)
        return list(map(int, chosen))

    def repair_playlist(self, cids: list[int], reorder: bool = False) -> list[int]:
        repaired = []
        seen = set()

        for cid in cids:
            cid = int(cid)
            if 0 <= cid < len(self.candidate_pool) and cid not in seen:
                repaired.append(cid)
                seen.add(cid)

        if len(repaired) < self.playlist_size:
            available = self.candidate_pool[~self.candidate_pool["cid"].isin(seen)]
            available = available.sort_values(["individual_mood_fit", "popularity"], ascending=False)
            for cid in available["cid"].values:
                if len(repaired) >= self.playlist_size:
                    break
                repaired.append(int(cid))
                seen.add(int(cid))

        repaired = repaired[:self.playlist_size]

        if reorder:
            repaired = self.greedy_order(repaired)

        return repaired

    def apply_operation(self, playlist: list[int], operation: dict[str, Any]) -> list[int]:
        new_playlist = list(map(int, playlist))
        action = operation.get("action")

        if action == "replace":
            pos = operation.get("position")
            new_cid = operation.get("new_cid")
            if pos is None or new_cid is None:
                return new_playlist
            try:
                pos = int(pos) - 1
                new_cid = int(new_cid)
            except Exception:
                return new_playlist
            if 0 <= pos < len(new_playlist) and 0 <= new_cid < len(self.candidate_pool):
                if new_cid not in new_playlist:
                    new_playlist[pos] = new_cid

        elif action == "move":
            from_pos = operation.get("from_position")
            to_pos = operation.get("to_position")
            if from_pos is None or to_pos is None:
                return new_playlist
            try:
                from_pos = int(from_pos) - 1
                to_pos = int(to_pos) - 1
            except Exception:
                return new_playlist
            if 0 <= from_pos < len(new_playlist) and 0 <= to_pos < len(new_playlist):
                item = new_playlist.pop(from_pos)
                new_playlist.insert(to_pos, item)

        return self.repair_playlist(new_playlist, reorder=False)

    def show_playlist(self, cids: list[int]) -> pd.DataFrame:
        pl = self.candidate_pool.iloc[list(map(int, cids))].copy().reset_index(drop=True)
        pl.insert(0, "position", np.arange(1, len(pl) + 1))
        cols = [
            "position", "cid", "track_name", "artists", "genre", "popularity",
            "duration_min", "danceability", "energy", "valence", "tempo",
            "individual_mood_fit"
        ]
        return pl[cols].round(3)

    def compact_playlist(self, cids: list[int]) -> list[dict[str, Any]]:
        rows = []
        pl = self.candidate_pool.iloc[list(map(int, cids))].reset_index(drop=True)
        for i, row in pl.iterrows():
            rows.append({
                "position": i + 1,
                "cid": int(row["cid"]),
                "track": str(row["track_name"])[:42],
                "artist": str(row["artists"])[:32],
                "genre": str(row["genre"])[:24],
                "popularity": int(row["popularity"]),
                "duration_min": round(float(row["duration_min"]), 2),
                "danceability": round(float(row["danceability"]), 3),
                "energy": round(float(row["energy"]), 3),
                "valence": round(float(row["valence"]), 3),
                "tempo": round(float(row["tempo"]), 1),
                "mood_fit": round(float(row["individual_mood_fit"]), 3),
            })
        return rows

    def generate_operation_menu(self, playlist: list[int], max_ops: int = 30) -> list[dict[str, Any]]:
        """Generate feasible mutation candidates and score them.

        The LLM then selects operation IDs from this menu.
        This makes Ollama-based optimization much more stable:
        the LLM still acts as the optimizer/selector, while the Python code
        checks objective function improvement objectively.
        """
        current_score = self.score_playlist(playlist)
        playlist_set = set(map(int, playlist))

        pl = self.candidate_pool.iloc[playlist].copy().reset_index(drop=True)

        # Weak positions: low mood fit and positions near weak transitions.
        weak_positions = set()

        for idx in np.argsort(pl["individual_mood_fit"].values)[:7]:
            weak_positions.add(int(idx) + 1)

        if len(pl) > 1:
            transition_scores = []
            for i in range(len(pl) - 1):
                transition_scores.append(self.transition_score(pl.iloc[i], pl.iloc[i + 1]))
            for idx in np.argsort(transition_scores)[:7]:
                weak_positions.add(int(idx) + 1)
                weak_positions.add(int(idx) + 2)

        # Replacement candidates: strong tracks not currently in playlist.
        available = self.candidate_pool[~self.candidate_pool["cid"].isin(playlist_set)].copy()
        available["replacement_priority"] = (
            0.70 * available["individual_mood_fit"]
            + 0.20 * (available["popularity"].clip(0, 100) / 100)
            + 0.10 * (1 - abs(available["duration_min"] - self.target_duration_min / self.playlist_size) / 4).clip(0, 1)
        )
        top_replacements = available.sort_values("replacement_priority", ascending=False).head(35)

        ops = []

        for pos in sorted(weak_positions):
            for _, cand in top_replacements.iterrows():
                op = {
                    "action": "replace",
                    "position": int(pos),
                    "new_cid": int(cand["cid"]),
                    "reason": "replace a weak track with a stronger candidate",
                }
                new_playlist = self.apply_operation(playlist, op)
                predicted_score = self.score_playlist(new_playlist)
                ops.append({
                    "operation_id": len(ops) + 1,
                    "operation": op,
                    "predicted_score": round(float(predicted_score), 6),
                    "predicted_delta": round(float(predicted_score - current_score), 6),
                })

        # Move candidates: try moving tracks around to improve smoothness.
        for from_pos in range(1, self.playlist_size + 1):
            for to_pos in range(1, self.playlist_size + 1):
                if from_pos == to_pos:
                    continue
                # Keep the menu compact.
                if abs(from_pos - to_pos) < 3:
                    continue
                op = {
                    "action": "move",
                    "from_position": int(from_pos),
                    "to_position": int(to_pos),
                    "reason": "reorder tracks to improve transition smoothness",
                }
                new_playlist = self.apply_operation(playlist, op)
                predicted_score = self.score_playlist(new_playlist)
                ops.append({
                    "operation_id": len(ops) + 1,
                    "operation": op,
                    "predicted_score": round(float(predicted_score), 6),
                    "predicted_delta": round(float(predicted_score - current_score), 6),
                })

        # Keep only the best candidate operations. Some can still have negative delta,
        # but with random start there are usually many positive ones.
        ops = sorted(ops, key=lambda x: x["predicted_delta"], reverse=True)
        return ops[:max_ops]

    @staticmethod
    def extract_json(text: str) -> dict[str, Any] | None:
        if text is None:
            return None

        text = str(text).strip()
        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        try:
            return json.loads(text)
        except Exception:
            pass

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None

        return None


def check_ollama(base_url: str) -> None:
    url = base_url.rstrip("/") + "/api/tags"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
    except Exception as exc:
        raise RuntimeError(
            "Cannot connect to Ollama.\n"
            "Open the Ollama app or run: ollama serve\n"
            f"Tried URL: {url}\n"
            f"Original error: {exc}"
        )


def call_ollama(prompt: str, model: str, base_url: str, temperature: float = 0.1) -> tuple[dict[str, Any] | None, str, str]:
    url = base_url.rstrip("/") + "/api/chat"

    messages = [
        {"role": "system", "content": "You are a strict JSON-producing optimizer. Return only valid JSON."},
        {"role": "user", "content": prompt},
    ]

    payloads = [
        {
            "mode": "schema",
            "payload": {
                "model": model,
                "messages": messages,
                "stream": False,
                "format": {
                    "type": "object",
                    "properties": {
                        "selected_operation_ids": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 1,
                            "maxItems": 3,
                        },
                        "reason": {"type": "string"},
                    },
                    "required": ["selected_operation_ids", "reason"],
                },
                "options": {"temperature": temperature},
            },
        },
        {
            "mode": "json",
            "payload": {
                "model": model,
                "messages": messages,
                "stream": False,
                "format": "json",
                "options": {"temperature": temperature},
            },
        },
        {
            "mode": "plain",
            "payload": {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature},
            },
        },
    ]

    last_text = ""

    for item in payloads:
        try:
            r = requests.post(url, json=item["payload"], timeout=180)
            r.raise_for_status()
            raw = r.json().get("message", {}).get("content", "")
            parsed = PlaylistProblem.extract_json(raw)
            if parsed is not None:
                return parsed, raw, item["mode"]
            last_text = raw
        except Exception as exc:
            last_text = repr(exc)

    return None, last_text, "failed"


def build_llm_prompt(problem: PlaylistProblem, playlist: list[int], operation_menu: list[dict[str, Any]]) -> str:
    current_score = problem.score_playlist(playlist, detailed=True)

    compact_menu = []
    for op_item in operation_menu:
        op = dict(op_item)
        compact_menu.append({
            "operation_id": op["operation_id"],
            "operation": op["operation"],
            "predicted_score": op["predicted_score"],
            "predicted_delta": op["predicted_delta"],
        })

    prompt = f"""
You are a local LLM used as an optimizer for a playlist optimization problem.

Current playlist:
{json.dumps(problem.compact_playlist(playlist), ensure_ascii=False)}

Current score:
{json.dumps(current_score, ensure_ascii=False)}

Objective:
Maximize total score. Higher predicted_delta is usually better, but also consider playlist logic:
- improve mood fit for scenario {problem.scenario}
- improve smooth transitions
- keep diversity
- keep duration close to {problem.target_duration_min} minutes

Available mutation operations:
{json.dumps(compact_menu, ensure_ascii=False)}

Task:
Select 1 to 3 operation IDs that are most promising.

Return only valid JSON:
{{
  "selected_operation_ids": [1, 2],
  "reason": "short explanation"
}}
""".strip()

    return prompt


def run_optimization(args: argparse.Namespace) -> None:
    check_ollama(args.ollama_url)

    problem = PlaylistProblem(
        dataset_path=args.dataset,
        scenario=args.scenario,
        playlist_size=args.playlist_size,
        candidate_pool_size=args.candidate_pool_size,
        target_duration_min=args.target_duration_min,
        seed=args.seed,
    )

    current = problem.build_initial_playlist(initial=args.initial)
    best = current.copy()
    best_score = problem.score_playlist(best)

    history = []

    print("\nProblem loaded")
    print(f"Rows after cleaning: {len(problem.data)}")
    print(f"Candidate pool size: {len(problem.candidate_pool)}")
    print(f"Initial mode: {args.initial}")
    print(f"Initial score: {best_score:.6f}")
    print(json.dumps(problem.score_playlist(best, detailed=True), ensure_ascii=False, indent=2))

    for iteration in tqdm(range(1, args.iterations + 1), desc="Ollama LLM optimizer"):
        before = problem.score_playlist(current, detailed=True)

        menu = problem.generate_operation_menu(current, max_ops=args.operation_menu_size)
        prompt = build_llm_prompt(problem, current, menu)

        parsed, raw, mode = call_ollama(
            prompt=prompt,
            model=args.model,
            base_url=args.ollama_url,
            temperature=args.temperature,
        )

        selected_ids = []
        if parsed and isinstance(parsed.get("selected_operation_ids"), list):
            selected_ids = [int(x) for x in parsed["selected_operation_ids"] if str(x).isdigit()]

        menu_by_id = {op["operation_id"]: op for op in menu}

        # If LLM returns invalid selection, use the best operation from the LLM-visible menu.
        # This backup is logged separately.
        if not selected_ids or not any(x in menu_by_id for x in selected_ids):
            selected_ids = [menu[0]["operation_id"]]
            mode = "backup_menu_top1"

        candidate = current.copy()
        applied_ops = []

        for op_id in selected_ids[:3]:
            if op_id not in menu_by_id:
                continue
            op = menu_by_id[op_id]["operation"]
            candidate = problem.apply_operation(candidate, op)
            applied_ops.append(op)

        # Optional local reorder after LLM-selected operations.
        # This improves smoothness and makes the graph clearer while preserving LLM-selected mutation.
        if args.reorder_after_llm:
            candidate = problem.greedy_order(candidate)

        after = problem.score_playlist(candidate, detailed=True)
        accepted = after["score"] > before["score"]

        if accepted:
            current = candidate

        current_score = problem.score_playlist(current)
        if current_score > best_score:
            best = current.copy()
            best_score = current_score

        history.append({
            "iteration": iteration,
            "mode": mode,
            "accepted": accepted,
            "before_score": before["score"],
            "proposal_score": after["score"],
            "current_score": current_score,
            "best_score": best_score,
            "selected_operation_ids": json.dumps(selected_ids, ensure_ascii=False),
            "applied_operations": json.dumps(applied_ops, ensure_ascii=False),
            "llm_raw_response": str(raw)[:2000],
            "before_mood_fit": before["mood_fit"],
            "before_smoothness": before["smoothness"],
            "before_diversity": before["diversity"],
            "before_popularity": before["popularity"],
            "before_duration_fit": before["duration_fit"],
            "proposal_mood_fit": after["mood_fit"],
            "proposal_smoothness": after["smoothness"],
            "proposal_diversity": after["diversity"],
            "proposal_popularity": after["popularity"],
            "proposal_duration_fit": after["duration_fit"],
        })

        print(
            f"Iter {iteration:02d} | mode={mode} | "
            f"before={before['score']:.4f} | proposal={after['score']:.4f} | "
            f"accepted={accepted} | best={best_score:.4f}"
        )

    save_outputs(problem, best, pd.DataFrame(history), args.output_dir)


def save_outputs(problem: PlaylistProblem, best: list[int], history: pd.DataFrame, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    final_playlist = problem.show_playlist(best)
    final_score = problem.score_playlist(best, detailed=True)

    final_playlist.to_csv(out / "optimized_playlist.csv", index=False)
    history.to_csv(out / "optimizer_history.csv", index=False)

    with open(out / "final_score.json", "w", encoding="utf-8") as f:
        json.dump(final_score, f, ensure_ascii=False, indent=2)

    initial_score = float(history.iloc[0]["before_score"]) if len(history) else final_score["score"]

    summary = f"""
Improved Ollama LLM as optimizer

Initial score: {initial_score:.6f}
Final score: {final_score['score']:.6f}
Improvement: {final_score['score'] - initial_score:.6f}

Final components:
{json.dumps(final_score, ensure_ascii=False, indent=2)}

Accepted iterations: {int(history['accepted'].sum()) if len(history) else 0}
Total iterations: {len(history)}

Interpretation:
The local LLM was used as an iterative optimizer. At each iteration it selected
promising mutation operations from a menu generated from the current playlist.
The Python code then applied the selected operations and accepted the new playlist
only if the formal objective score improved.
""".strip()

    with open(out / "summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)

    plt.figure(figsize=(10, 5))
    plt.plot(history["iteration"], history["before_score"], marker="o", label="Before proposal")
    plt.plot(history["iteration"], history["proposal_score"], marker="o", label="LLM proposal")
    plt.plot(history["iteration"], history["best_score"], marker="o", label="Best score")
    plt.xlabel("Iteration")
    plt.ylabel("Playlist Score")
    plt.title("Improved Ollama LLM as optimizer: convergence")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out / "convergence.png", dpi=200)
    plt.close()

    component_names = ["mood_fit", "smoothness", "diversity", "popularity", "duration_fit"]
    initial_components = {
        "mood_fit": float(history.iloc[0]["before_mood_fit"]) if len(history) else final_score["mood_fit"],
        "smoothness": float(history.iloc[0]["before_smoothness"]) if len(history) else final_score["smoothness"],
        "diversity": float(history.iloc[0]["before_diversity"]) if len(history) else final_score["diversity"],
        "popularity": float(history.iloc[0]["before_popularity"]) if len(history) else final_score["popularity"],
        "duration_fit": float(history.iloc[0]["before_duration_fit"]) if len(history) else final_score["duration_fit"],
    }

    comparison = pd.DataFrame({
        "component": component_names,
        "initial": [initial_components[c] for c in component_names],
        "final": [final_score[c] for c in component_names],
    })
    comparison.to_csv(out / "component_comparison.csv", index=False)

    comparison.plot(x="component", y=["initial", "final"], kind="bar", figsize=(10, 5))
    plt.ylabel("Value")
    plt.title("Initial vs final playlist components")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(out / "components_comparison.png", dpi=200)
    plt.close()

    print("\nFinal score:")
    print(json.dumps(final_score, ensure_ascii=False, indent=2))

    print("\nSaved files:")
    for p in sorted(out.iterdir()):
        print("-", p)

    print("\nFinal playlist:")
    print(final_playlist.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Improved Ollama LLM as optimizer for Spotify playlist.")

    parser.add_argument("--dataset", default="dataset.csv")
    parser.add_argument("--model", default="qwen2.5:0.5b")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--scenario", default="party", choices=list(TARGET_PROFILES.keys()))
    parser.add_argument("--playlist-size", type=int, default=20)
    parser.add_argument("--candidate-pool-size", type=int, default=800)
    parser.add_argument("--operation-menu-size", type=int, default=30)
    parser.add_argument("--target-duration-min", type=float, default=60)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial", choices=["random", "greedy"], default="random")
    parser.add_argument("--output-dir", default="outputs_improved")
    parser.add_argument("--reorder-after-llm", action="store_true", default=True)

    return parser.parse_args()


if __name__ == "__main__":
    try:
        start = time.time()
        run_optimization(parse_args())
        print(f"\nElapsed time: {time.time() - start:.2f} seconds")
    except Exception as exc:
        print("\nERROR:")
        print(exc)
        sys.exit(1)
