"""
Tabu Search for Spotify Playlist Optimization
Запуск: python tabu_search.py
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import deque
from pathlib import Path
from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================================
# Целевые профили и веса
# ============================================================================

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


# ============================================================================
# Класс задачи
# ============================================================================

class PlaylistProblem:
    def __init__(
        self,
        dataset_path: str = "data/dataset.csv",
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
                f"Dataset file not found: {self.dataset_path}\n"
                f"Make sure dataset.csv is in the same folder as this script"
            )

        print(f"Loading dataset from {self.dataset_path}...")
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
        
        print(f"  Loaded {len(data)} tracks after cleaning")
        return data

    def build_candidate_pool(self) -> pd.DataFrame:
        pool = (
            self.data.sort_values(["individual_mood_fit", "popularity"], ascending=False)
            .head(self.candidate_pool_size)
            .copy()
            .reset_index(drop=True)
        )
        pool["cid"] = pool.index
        print(f"  Candidate pool: {len(pool)} tracks")
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

    def greedy_order(self, cids: list[int]) -> list[int]:
        cids = list(map(int, cids))
        if not cids:
            return []

        start = min(
            cids,
            key=lambda cid: abs(
                self.candidate_pool.iloc[cid]["tempo"] - self.target_profile["tempo"]
            )
        )
        ordered = [start]
        remaining = set(cids)
        remaining.remove(start)

        while remaining:
            prev = ordered[-1]
            nxt = max(
                remaining,
                key=lambda cid: 0.65 * self.transition_score(
                    self.candidate_pool.iloc[prev],
                    self.candidate_pool.iloc[cid]
                ) + 0.35 * float(self.candidate_pool.iloc[cid]["individual_mood_fit"])
            )
            ordered.append(int(nxt))
            remaining.remove(nxt)

        return ordered

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
            available = available.sort_values(
                ["individual_mood_fit", "popularity"], ascending=False
            )
            for cid in available["cid"].values:
                if len(repaired) >= self.playlist_size:
                    break
                repaired.append(int(cid))
                seen.add(int(cid))

        repaired = repaired[:self.playlist_size]

        if reorder:
            repaired = self.greedy_order(repaired)

        return repaired

    def show_playlist(self, cids: list[int]) -> pd.DataFrame:
        pl = self.candidate_pool.iloc[list(map(int, cids))].copy().reset_index(drop=True)
        pl.insert(0, "position", np.arange(1, len(pl) + 1))
        cols = [
            "position", "cid", "track_name", "artists", "genre", "popularity",
            "duration_min", "danceability", "energy", "valence", "tempo",
            "individual_mood_fit"
        ]
        return pl[cols].round(3)


# ============================================================================
# TABU SEARCH ALGORITHM
# ============================================================================

class TabuSearchOptimizer:
    def __init__(
        self,
        problem: PlaylistProblem,
        tabu_tenure: int = 15,
        max_iterations: int = 300,
        neighbor_size: int = 50,
        aspiration_criterion: bool = True,
        diversification_frequency: int = 100,
        intensification_frequency: int = 50,
        seed: int = 42,
    ):
        self.problem = problem
        self.tabu_tenure = tabu_tenure
        self.max_iterations = max_iterations
        self.neighbor_size = neighbor_size
        self.aspiration_criterion = aspiration_criterion
        self.diversification_frequency = diversification_frequency
        self.intensification_frequency = intensification_frequency

        random.seed(seed)
        np.random.seed(seed)

        self.all_cids = self.problem.candidate_pool["cid"].tolist()
        self.pool_size = len(self.all_cids)

        self.current_solution: list[int] = []
        self.current_score: float = 0.0
        self.best_solution: list[int] = []
        self.best_score: float = 0.0

        self.tabu_list: deque[Tuple[int, int, int, int]] = deque(maxlen=tabu_tenure * 2)
        self.frequency_memory: dict[Tuple[int, int], int] = {}
        self.history: list[dict[str, Any]] = []

    def generate_random_playlist(self) -> list[int]:
        return random.sample(self.all_cids, self.problem.playlist_size)

    def get_neighbor_moves(self, solution: list[int]) -> list[Tuple[list[int], Tuple[int, int, int]]]:
        neighbors = []
        size = self.problem.playlist_size
        
        positions_to_try = random.sample(range(size), min(size, 5))
        
        for pos in positions_to_try:
            old_track = solution[pos]
            current_tracks = set(solution)
            candidates = [cid for cid in self.all_cids if cid not in current_tracks]
            
            if not candidates:
                continue
            
            sample_size = min(len(candidates), self.neighbor_size // len(positions_to_try) + 1)
            candidates_sample = random.sample(candidates, sample_size)
            
            for new_track in candidates_sample:
                new_solution = solution[:]
                new_solution[pos] = new_track
                neighbors.append((new_solution, (pos, old_track, new_track)))
        
        random.shuffle(neighbors)
        return neighbors[:self.neighbor_size]

    def is_tabu(self, move: Tuple[int, int, int], iteration: int) -> bool:
        pos, old_track, new_track = move
        
        for tabu_pos, tabu_old, tabu_new, end_iter in self.tabu_list:
            if tabu_pos == pos and tabu_new == new_track and tabu_old == old_track:
                return True
            if tabu_pos == pos and tabu_new == old_track and tabu_old == new_track:
                return True
        
        return False

    def add_to_tabu(self, move: Tuple[int, int, int], iteration: int):
        pos, old_track, new_track = move
        end_iteration = iteration + self.tabu_tenure
        self.tabu_list.append((pos, old_track, new_track, end_iteration))
        
        key = (pos, old_track)
        self.frequency_memory[key] = self.frequency_memory.get(key, 0) + 1

    def diversify(self) -> list[int]:
        solution = self.current_solution[:]
        size = self.problem.playlist_size
        
        num_changes = random.randint(size // 3, size // 2)
        positions = random.sample(range(size), num_changes)
        
        for pos in positions:
            old_track = solution[pos]
            current_set = set(solution)
            
            candidates = []
            for cid in self.all_cids:
                if cid not in current_set:
                    key = (pos, cid)
                    frequency = self.frequency_memory.get(key, 0)
                    candidates.append((cid, frequency))
            
            if candidates:
                candidates.sort(key=lambda x: x[1])
                best_candidates = [c[0] for c in candidates[:min(20, len(candidates))]]
                new_track = random.choice(best_candidates) if best_candidates else random.choice(self.all_cids)
                while new_track in current_set:
                    new_track = random.choice(self.all_cids)
                solution[pos] = new_track
                current_set.remove(old_track)
                current_set.add(new_track)
        
        return solution

    def intensify(self) -> list[int]:
        if not self.best_solution:
            return self.current_solution[:]
        
        solution = self.best_solution[:]
        size = self.problem.playlist_size
        
        num_changes = random.randint(1, 2)
        positions = random.sample(range(size), num_changes)
        
        for pos in positions:
            old_track = solution[pos]
            current_set = set(solution)
            candidates = [cid for cid in self.all_cids if cid not in current_set]
            
            if candidates:
                candidates_with_fit = [(cid, self.problem.candidate_pool.iloc[cid]["individual_mood_fit"]) 
                                       for cid in candidates]
                candidates_with_fit.sort(key=lambda x: x[1], reverse=True)
                best_candidates = [c[0] for c in candidates_with_fit[:min(10, len(candidates_with_fit))]]
                new_track = random.choice(best_candidates) if best_candidates else random.choice(candidates)
                solution[pos] = new_track
        
        return solution

    def run(self, verbose: bool = True) -> dict[str, Any]:
        print("\nInitializing Tabu Search...")
        self.current_solution = self.generate_random_playlist()
        self.current_solution = self.problem.repair_playlist(self.current_solution, reorder=True)
        self.current_score = self.problem.score_playlist(self.current_solution)
        self.best_solution = self.current_solution[:]
        self.best_score = self.current_score
        
        self.tabu_list.clear()
        self.frequency_memory.clear()
        self.history = []
        
        stagnation_counter = 0
        
        print(f"Running for {self.max_iterations} iterations...\n")
        pbar = tqdm(range(self.max_iterations), desc="Tabu Search", disable=not verbose)
        
        for iteration in pbar:
            neighbors = self.get_neighbor_moves(self.current_solution)
            
            if not neighbors:
                self.current_solution = self.diversify()
                self.current_solution = self.problem.repair_playlist(self.current_solution, reorder=True)
                self.current_score = self.problem.score_playlist(self.current_solution)
                continue
            
            best_neighbor = None
            best_neighbor_score = -float('inf')
            best_move = None
            
            for neighbor_solution, move in neighbors:
                neighbor_solution = self.problem.repair_playlist(neighbor_solution)
                neighbor_score = self.problem.score_playlist(neighbor_solution)
                
                is_tabu = self.is_tabu(move, iteration)
                
                if is_tabu and self.aspiration_criterion:
                    if neighbor_score > self.best_score:
                        is_tabu = False
                
                if not is_tabu:
                    if neighbor_score > best_neighbor_score:
                        best_neighbor_score = neighbor_score
                        best_neighbor = neighbor_solution
                        best_move = move
            
            if best_neighbor is None and neighbors:
                best_idx = max(range(len(neighbors)), 
                              key=lambda i: self.problem.score_playlist(
                                  self.problem.repair_playlist(neighbors[i][0])
                              ))
                best_neighbor, best_move = neighbors[best_idx]
                best_neighbor = self.problem.repair_playlist(best_neighbor)
                best_neighbor_score = self.problem.score_playlist(best_neighbor)
            elif best_neighbor is None:
                best_neighbor = self.diversify()
                best_neighbor = self.problem.repair_playlist(best_neighbor, reorder=True)
                best_neighbor_score = self.problem.score_playlist(best_neighbor)
                best_move = None
            
            self.current_solution = best_neighbor
            self.current_score = best_neighbor_score
            
            if best_move:
                self.add_to_tabu(best_move, iteration)
            
            if self.current_score > self.best_score:
                self.best_score = self.current_score
                self.best_solution = self.current_solution[:]
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            if stagnation_counter > self.diversification_frequency:
                self.current_solution = self.diversify()
                self.current_solution = self.problem.repair_playlist(self.current_solution, reorder=True)
                self.current_score = self.problem.score_playlist(self.current_solution)
                stagnation_counter = 0
            
            if iteration > 0 and iteration % self.intensification_frequency == 0:
                if random.random() < 0.5:
                    self.current_solution = self.intensify()
                    self.current_solution = self.problem.repair_playlist(self.current_solution, reorder=True)
                    self.current_score = self.problem.score_playlist(self.current_solution)
            
            best_detailed = self.problem.score_playlist(self.best_solution, detailed=True)
            
            self.history.append({
                "iteration": iteration,
                "current_score": self.current_score,
                "best_score": self.best_score,
                "tabu_list_size": len(self.tabu_list),
                "best_mood_fit": best_detailed["mood_fit"],
                "best_smoothness": best_detailed["smoothness"],
                "best_diversity": best_detailed["diversity"],
                "best_popularity": best_detailed["popularity"],
                "best_duration_fit": best_detailed["duration_fit"],
            })
            
            if verbose:
                pbar.set_description(
                    f"Iter {iteration:03d} | Best: {self.best_score:.4f} | "
                    f"Current: {self.current_score:.4f}"
                )
        
        print("\nFinal greedy ordering...")
        self.best_solution = self.problem.greedy_order(self.best_solution)
        self.best_score = self.problem.score_playlist(self.best_solution)
        
        return {
            "best_playlist": self.best_solution,
            "best_score": self.best_score,
            "best_detailed": self.problem.score_playlist(self.best_solution, detailed=True),
            "history": self.history,
        }


# ============================================================================
# Сохранение результатов (расширенная версия)
# ============================================================================

def save_results(problem: PlaylistProblem, result: dict, output_dir: str = "outputs_tabu"):
    """Сохраняет результаты в файлы с расширенной визуализацией."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # ========== 1. Базовые сохранения ==========
    # Сохраняем плейлист
    playlist_df = problem.show_playlist(result["best_playlist"])
    playlist_df.to_csv(out / "optimized_playlist.csv", index=False)
    
    # Сохраняем историю
    history_df = pd.DataFrame(result["history"])
    history_df.to_csv(out / "optimizer_history.csv", index=False)
    
    # Сохраняем финальный скор
    with open(out / "final_score.json", "w", encoding="utf-8") as f:
        json.dump(result["best_detailed"], f, ensure_ascii=False, indent=2)
    
    # ========== 2. Улучшенный текстовый отчёт ==========
    initial_score = result["history"][0]["best_score"] if result["history"] else result["best_score"]
    improvement_pct = (result['best_score'] - initial_score) / initial_score * 100 if initial_score > 0 else 0
    
    summary = f"""Tabu Search for Spotify Playlist Optimization

Parameters:
- Tabu tenure: {15}
- Max iterations: {300}
- Neighbor size: {50}
- Aspiration criterion: True
- Diversification frequency: {100}
- Intensification frequency: {50}
- Scenario: party
- Seed: 42

Results:
- Initial best score: {initial_score:.6f}
- Final best score: {result['best_score']:.6f}
- Improvement: {result['best_score'] - initial_score:.6f}
- Improvement (%): {improvement_pct:.2f}%

Final components:
{json.dumps(result['best_detailed'], ensure_ascii=False, indent=2)}

Elapsed time: {result.get('elapsed_time', 0):.2f} seconds
"""

    with open(out / "summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)
    
    # ========== 3. График сходимости (оригинальный) ==========
    plt.figure(figsize=(12, 5))
    iterations = [h["iteration"] for h in result["history"]]
    best_scores = [h["best_score"] for h in result["history"]]
    current_scores = [h["current_score"] for h in result["history"]]
    
    plt.subplot(1, 2, 1)
    plt.plot(iterations, best_scores, 'b-', linewidth=2, label='Best score')
    plt.plot(iterations, current_scores, 'orange', linewidth=1, alpha=0.5, label='Current score')
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.title('Tabu Search Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    tabu_sizes = [h["tabu_list_size"] for h in result["history"]]
    plt.plot(iterations, tabu_sizes, 'green', linewidth=1)
    plt.xlabel('Iteration')
    plt.ylabel('Tabu List Size')
    plt.title('Tabu List Size Over Time')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out / "convergence.png", dpi=150)
    plt.close()
    
    # ========== 4. НОВЫЙ ГРАФИК: Эволюция компонент ==========
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Tabu Search Evolution of Components (scenario: party)', fontsize=14)

    generations = [h["iteration"] for h in result["history"]]
    
    components = ['best_mood_fit', 'best_smoothness', 'best_diversity', 'best_popularity', 'best_duration_fit']
    titles = ['Mood Fit (настроение)', 'Smoothness (плавность)', 
              'Diversity (разнообразие)', 'Popularity (популярность)', 
              'Duration Fit (длительность)']
    
    for idx, (comp, title) in enumerate(zip(components, titles)):
        ax = axes[idx // 3, idx % 3]
        values = [h[comp] for h in result["history"]]
        ax.plot(generations, values, 'b-', linewidth=1.5)
        ax.fill_between(generations, values, alpha=0.3)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(title)
        ax.set_title(f'{title}: {result["best_detailed"][comp.replace("best_", "")]:.4f}')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    axes[1, 2].axis('off')
    plt.tight_layout()
    plt.savefig(out / "components_evolution.png", dpi=150)
    plt.close()
    
    # ========== 5. НОВЫЙ ГРАФИК: Радарная диаграмма ==========
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

    components_list = ['mood_fit', 'smoothness', 'diversity', 'popularity', 'duration_fit']
    component_names = ['Mood\nFit', 'Smooth-\nness', 'Diver-\nsity', 'Popu-\nlarity', 'Duration\nFit']

    final_values = [result['best_detailed'][c] for c in components_list]
    initial_values = [result["history"][0][f'best_{c}'] for c in components_list] if result["history"] else final_values

    angles = np.linspace(0, 2 * np.pi, len(components_list), endpoint=False).tolist()
    angles += angles[:1]

    final_vals_plot = final_values + final_values[:1]
    ax.plot(angles, final_vals_plot, 'o-', linewidth=2, label='Final', color='blue')
    ax.fill(angles, final_vals_plot, alpha=0.25, color='blue')

    initial_vals_plot = initial_values + initial_values[:1]
    ax.plot(angles, initial_vals_plot, 'o-', linewidth=2, label='Initial', color='gray', alpha=0.7)
    ax.fill(angles, initial_vals_plot, alpha=0.1, color='gray')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(component_names, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.set_title('Playlist Quality Components (Radar Chart)', fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)

    plt.savefig(out / "radar_chart.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # ========== 6. НОВЫЙ ГРАФИК: Гистограмма улучшений ==========
    plt.figure(figsize=(10, 5))

    improvements = []
    for i in range(1, len(result["history"])):
        improvements.append(result["history"][i]['best_score'] - result["history"][i-1]['best_score'])

    plt.hist(improvements, bins=30, color='green', alpha=0.7, edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', label='No improvement')
    plt.xlabel('Improvement per Iteration')
    plt.ylabel('Frequency')
    plt.title('Distribution of Improvements Across Iterations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out / "improvements_histogram.png", dpi=150)
    plt.close()
    
    # ========== 7. НОВЫЙ ГРАФИК: Тепловая карта жанров ==========
    playlist_cids = result["best_playlist"]
    playlist_tracks = problem.candidate_pool.iloc[playlist_cids]
    genres = playlist_tracks['genre'].tolist()
    
    unique_genres = list(set(genres))
    genre_matrix = np.zeros((len(genres), len(unique_genres)))
    for i, genre in enumerate(genres):
        genre_matrix[i, unique_genres.index(genre)] = 1
    
    plt.figure(figsize=(12, 6))
    plt.imshow(genre_matrix, aspect='auto', cmap='YlOrRd')
    plt.xlabel('Genre')
    plt.ylabel('Track Position')
    plt.title('Genre Distribution in Optimized Playlist')
    plt.colorbar(label='Genre presence')
    
    plt.xticks(range(len(unique_genres)), unique_genres, rotation=45, ha='right', fontsize=8)
    plt.yticks(range(len(genres)), [f"#{i+1}" for i in range(len(genres))], fontsize=8)
    
    plt.tight_layout()
    plt.savefig(out / "genre_heatmap.png", dpi=150)
    plt.close()
    
    # ========== 8. НОВЫЙ ГРАФИК: Топ треков ==========
    plt.figure(figsize=(12, 6))
    
    playlist_with_scores = []
    for idx, cid in enumerate(playlist_cids):
        track = problem.candidate_pool.iloc[cid]
        playlist_with_scores.append({
            'position': idx + 1,
            'name': track['track_name'][:30],
            'mood_fit': track['individual_mood_fit']
        })
    
    top_tracks = sorted(playlist_with_scores, key=lambda x: x['mood_fit'], reverse=True)[:10]
    
    names = [t['name'] for t in top_tracks]
    moods = [t['mood_fit'] for t in top_tracks]
    
    bars = plt.barh(names, moods, color='skyblue', edgecolor='navy')
    plt.xlabel('Individual Mood Fit Score')
    plt.title('Top 10 Tracks by Individual Mood Fit')
    plt.xlim(0, 1)
    plt.grid(axis='x', alpha=0.3)
    
    for bar, mood in zip(bars, moods):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{mood:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(out / "top_tracks_mood_fit.png", dpi=150)
    plt.close()
    
    print(f"\n✅ Results saved to '{output_dir}/'")
    print(f"   Files generated:")
    for p in sorted(out.iterdir()):
        print(f"     - {p.name}")
    return out


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("TABU SEARCH FOR SPOTIFY PLAYLIST OPTIMIZATION")
    print("=" * 60)
    
    # ===== НАСТРОЙКИ =====
    DATASET_PATH = "data/dataset.csv"
    SCENARIO = "party"
    PLAYLIST_SIZE = 20
    CANDIDATE_POOL_SIZE = 800
    TARGET_DURATION_MIN = 60.0
    
    TABU_TENURE = 15
    MAX_ITERATIONS = 300
    NEIGHBOR_SIZE = 50
    ASPIRATION = True
    DIVERSIFICATION_FREQ = 100
    INTENSIFICATION_FREQ = 50
    SEED = 42
    # ====================
    
    print(f"\nConfiguration:")
    print(f"  Dataset: {DATASET_PATH}")
    print(f"  Scenario: {SCENARIO}")
    print(f"  Playlist size: {PLAYLIST_SIZE}")
    print(f"  Candidate pool: {CANDIDATE_POOL_SIZE}")
    print(f"  Target duration: {TARGET_DURATION_MIN} min")
    print(f"  Tabu tenure: {TABU_TENURE}")
    print(f"  Max iterations: {MAX_ITERATIONS}")
    print(f"  Neighbor size: {NEIGHBOR_SIZE}")
    print(f"  Aspiration: {ASPIRATION}")
    print(f"  Seed: {SEED}")
    
    start_time = time.time()
    
    # Создаём задачу
    problem = PlaylistProblem(
        dataset_path=DATASET_PATH,
        scenario=SCENARIO,
        playlist_size=PLAYLIST_SIZE,
        candidate_pool_size=CANDIDATE_POOL_SIZE,
        target_duration_min=TARGET_DURATION_MIN,
        seed=SEED,
    )
    
    # Создаём и запускаем оптимизатор
    ts = TabuSearchOptimizer(
        problem=problem,
        tabu_tenure=TABU_TENURE,
        max_iterations=MAX_ITERATIONS,
        neighbor_size=NEIGHBOR_SIZE,
        aspiration_criterion=ASPIRATION,
        diversification_frequency=DIVERSIFICATION_FREQ,
        intensification_frequency=INTENSIFICATION_FREQ,
        seed=SEED,
    )
    
    result = ts.run(verbose=True)
    
    elapsed = time.time() - start_time
    result['elapsed_time'] = elapsed
    
    # Выводим результаты
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nFinal score: {result['best_score']:.6f}")
    print(f"Time: {elapsed:.2f} seconds")
    
    print("\nFinal components:")
    for k, v in result['best_detailed'].items():
        print(f"  {k}: {v:.4f}")
    
    # Показываем плейлист
    print("\n" + "=" * 60)
    print("FINAL PLAYLIST (first 10 tracks)")
    print("=" * 60)
    playlist_df = problem.show_playlist(result["best_playlist"])
    display_cols = ['position', 'track_name', 'artists', 'genre', 'duration_min', 'individual_mood_fit']
    print(playlist_df[display_cols].head(10).to_string(index=False))
    
    # Сохраняем результаты
    save_results(problem, result)
    
    print(f"\n✅ Done! Check the 'outputs_tabu' folder for results.")


if __name__ == "__main__":
    main()