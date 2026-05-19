"""
Genetic Algorithm for Spotify Playlist Optimization.
Сравнивается с LLM-оптимизатором на той же задаче и с той же функцией оценки.

Run example:
    python genetic_playlist_optimizer.py --dataset dataset.csv --scenario party --generations 100
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================================
# Целевые профили и веса — те же, что в LLM-оптимизаторе
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
# Класс задачи — идентичен LLM-версии для честного сравнения
# ============================================================================

class PlaylistProblem:
    """
    Задача оптимизации плейлиста.
    Использует ту же функцию оценки, что и LLM-оптимизатор,
    чтобы сравнение было максимально честным.
    """

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

    def greedy_order(self, cids: list[int]) -> list[int]:
        """Жадное упорядочивание треков для улучшения плавности переходов."""
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
        """Исправляет плейлист: удаляет дубли, дополняет до нужного размера."""
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
# Генетический алгоритм
# ============================================================================

class GeneticPlaylistOptimizer:
    """
    Генетический алгоритм для оптимизации плейлиста.

    Основные параметры:
    - population_size: размер популяции (сколько плейлистов живёт одновременно)
    - generations: количество поколений
    - mutation_rate: вероятность мутации одной особи
    - mutation_strength: сколько треков заменить при мутации (доля от размера плейлиста)
    - crossover_rate: вероятность скрещивания (иначе особь просто копируется)
    - elitism_count: сколько лучших особей гарантированно переходит в следующее поколение
    - tournament_size: сколько особей участвуют в турнирной селекции
    """

    def __init__(
        self,
        problem: PlaylistProblem,
        population_size: int = 100,
        generations: int = 100,
        mutation_rate: float = 0.15,
        mutation_strength: float = 0.15,
        crossover_rate: float = 0.85,
        elitism_count: int = 5,
        tournament_size: int = 5,
        seed: int = 42,
    ):
        self.problem = problem
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.tournament_size = tournament_size

        random.seed(seed)
        np.random.seed(seed)

        # Инициализация популяции
        self.population: list[list[int]] = []
        self.fitness_scores: list[float] = []
        self.best_individual: list[int] = []
        self.best_score: float = 0.0
        self.history: list[dict[str, Any]] = []

        # Все доступные ID треков (для быстрой генерации случайных)
        self.all_cids = self.problem.candidate_pool["cid"].tolist()
        self.pool_size = len(self.all_cids)

    def initialize_population(self):
        """Создаёт начальную популяцию случайных плейлистов."""
        self.population = []
        for _ in range(self.population_size):
            # Случайно выбираем playlist_size треков без повторений
            individual = random.sample(self.all_cids, self.problem.playlist_size)
            self.population.append(individual)

    def evaluate_population(self) -> list[float]:
        """Считает скор для каждой особи в популяции."""
        return [self.problem.score_playlist(ind) for ind in self.population]

    def tournament_selection(self, fitness: list[float]) -> list[int]:
        """
        Турнирная селекция: выбирает tournament_size случайных особей
        и возвращает лучшую из них.
        """
        tournament_indices = random.sample(range(self.population_size), self.tournament_size)
        best_idx = max(tournament_indices, key=lambda i: fitness[i])
        return self.population[best_idx][:]

    def crossover(self, parent1: list[int], parent2: list[int]) -> list[int]:
        """
        Одноточечный кроссовер: берём начало от parent1, конец от parent2.
        Затем удаляем дубликаты и дополняем до нужного размера.
        """
        size = self.problem.playlist_size
        crossover_point = random.randint(1, size - 1)

        child = parent1[:crossover_point] + parent2[crossover_point:]

        # Удаляем дубликаты, сохраняя порядок
        seen = set()
        unique_child = []
        for cid in child:
            if cid not in seen:
                unique_child.append(cid)
                seen.add(cid)

        # Дополняем недостающими треками, если нужно
        if len(unique_child) < size:
            available = [cid for cid in self.all_cids if cid not in seen]
            random.shuffle(available)
            unique_child.extend(available[:size - len(unique_child)])

        return unique_child[:size]

    def mutate(self, individual: list[int]) -> list[int]:
        """
        Мутация: заменяет случайное количество треков на случайные из пула.
        Количество замен = mutation_strength * playlist_size.
        """
        num_mutations = max(1, int(self.mutation_strength * self.problem.playlist_size))
        mutated = individual[:]
        current_set = set(mutated)

        for _ in range(num_mutations):
            # Выбираем случайную позицию для замены
            pos = random.randint(0, len(mutated) - 1)
            # Выбираем новый трек, которого ещё нет в плейлисте
            available = [cid for cid in self.all_cids if cid not in current_set]
            if available:
                new_cid = random.choice(available)
                current_set.remove(mutated[pos])
                current_set.add(new_cid)
                mutated[pos] = new_cid

        return mutated

    def run(self, verbose: bool = True) -> dict[str, Any]:
        """
        Запускает генетический алгоритм и возвращает результаты.

        Returns:
            dict с лучшим плейлистом, скором и историей поколений.
        """
        # Шаг 1: начальная популяция
        self.initialize_population()
        self.fitness_scores = self.evaluate_population()

        # Находим лучшего в начальной популяции
        best_idx = max(range(self.population_size), key=lambda i: self.fitness_scores[i])
        self.best_individual = self.population[best_idx][:]
        self.best_score = self.fitness_scores[best_idx]

        self.history = []

        # Основной цикл по поколениям
        pbar = tqdm(range(self.generations), desc="Genetic Algorithm", disable=not verbose)

        for gen in pbar:
            # Сортируем популяцию по скору (для элитизма)
            sorted_indices = sorted(
                range(self.population_size),
                key=lambda i: self.fitness_scores[i],
                reverse=True
            )
            sorted_population = [self.population[i] for i in sorted_indices]
            sorted_fitness = [self.fitness_scores[i] for i in sorted_indices]

            # Обновляем лучшего
            if sorted_fitness[0] > self.best_score:
                self.best_individual = sorted_population[0][:]
                self.best_score = sorted_fitness[0]

            # Записываем историю
            avg_fitness = float(np.mean(self.fitness_scores))
            best_detailed = self.problem.score_playlist(self.best_individual, detailed=True)

            self.history.append({
                "generation": gen,
                "best_score": self.best_score,
                "avg_score": avg_fitness,
                "best_mood_fit": best_detailed["mood_fit"],
                "best_smoothness": best_detailed["smoothness"],
                "best_diversity": best_detailed["diversity"],
                "best_popularity": best_detailed["popularity"],
                "best_duration_fit": best_detailed["duration_fit"],
            })

            if verbose:
                pbar.set_description(
                    f"Gen {gen:03d} | Best: {self.best_score:.4f} | Avg: {avg_fitness:.4f}"
                )

            # Формируем новое поколение
            new_population = []

            # Элитизм: сохраняем лучших
            for i in range(min(self.elitism_count, self.population_size)):
                new_population.append(sorted_population[i][:])

            # Заполняем остальное потомками
            while len(new_population) < self.population_size:
                # Селекция родителей
                parent1 = self.tournament_selection(self.fitness_scores)
                parent2 = self.tournament_selection(self.fitness_scores)

                # Кроссовер
                if random.random() < self.crossover_rate:
                    child = self.crossover(parent1, parent2)
                else:
                    child = random.choice([parent1, parent2])[:]

                # Мутация
                if random.random() < self.mutation_rate:
                    child = self.mutate(child)

                new_population.append(child)

            # Обновляем популяцию
            self.population = new_population[:self.population_size]
            self.fitness_scores = self.evaluate_population()

        # Финальное обновление
        final_idx = max(range(self.population_size), key=lambda i: self.fitness_scores[i])
        if self.fitness_scores[final_idx] > self.best_score:
            self.best_individual = self.population[final_idx][:]
            self.best_score = self.fitness_scores[final_idx]

        return {
            "best_playlist": self.best_individual,
            "best_score": self.best_score,
            "best_detailed": self.problem.score_playlist(self.best_individual, detailed=True),
            "history": self.history,
        }


# ============================================================================
# Сохранение результатов
# ============================================================================

def save_results(
    problem: PlaylistProblem,
    result: dict[str, Any],
    history: list[dict[str, Any]],
    output_dir: str,
    elapsed_time: float,
    args: argparse.Namespace,
):
    """Сохраняет все результаты оптимизации."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Сохраняем финальный плейлист
    final_playlist_df = problem.show_playlist(result["best_playlist"])
    final_playlist_df.to_csv(out / "optimized_playlist.csv", index=False)

    # Сохраняем историю
    history_df = pd.DataFrame(history)
    history_df.to_csv(out / "optimizer_history.csv", index=False)

    # Сохраняем финальный скор
    with open(out / "final_score.json", "w", encoding="utf-8") as f:
        json.dump(result["best_detailed"], f, ensure_ascii=False, indent=2)

    # Сводка
    initial_score = history[0]["best_score"] if history else result["best_score"]
    summary = f"""
Genetic Algorithm for Spotify Playlist Optimization

Parameters:
- Population size: {args.population_size}
- Generations: {args.generations}
- Mutation rate: {args.mutation_rate}
- Crossover rate: {args.crossover_rate}
- Elitism count: {args.elitism_count}
- Tournament size: {args.tournament_size}
- Scenario: {args.scenario}

Initial best score: {initial_score:.6f}
Final best score: {result['best_score']:.6f}
Improvement: {result['best_score'] - initial_score:.6f}

Final components:
{json.dumps(result['best_detailed'], ensure_ascii=False, indent=2)}

Elapsed time: {elapsed_time:.2f} seconds
""".strip()

    with open(out / "summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)

    # График сходимости
    plt.figure(figsize=(10, 5))
    gens = [h["generation"] for h in history]
    best_scores = [h["best_score"] for h in history]
    avg_scores = [h["avg_score"] for h in history]

    plt.plot(gens, best_scores, "b-", linewidth=2, label="Best score")
    plt.plot(gens, avg_scores, "orange", linewidth=1, alpha=0.7, label="Average score")
    plt.xlabel("Generation")
    plt.ylabel("Playlist Score")
    plt.title(f"Genetic Algorithm Convergence (scenario: {args.scenario})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "convergence.png", dpi=200)
    plt.close()

    # Сравнение компонент (первое vs последнее поколение)
    component_names = ["mood_fit", "smoothness", "diversity", "popularity", "duration_fit"]
    initial_components = {
        c: history[0][f"best_{c}"] for c in component_names
    } if history else {c: result["best_detailed"][c] for c in component_names}
    final_components = {
        c: result["best_detailed"][c] for c in component_names
    }

    comparison = pd.DataFrame({
        "component": component_names,
        "initial": [initial_components[c] for c in component_names],
        "final": [final_components[c] for c in component_names],
    })
    comparison.to_csv(out / "component_comparison.csv", index=False)

    comparison.plot(x="component", y=["initial", "final"], kind="bar", figsize=(10, 5))
    plt.ylabel("Value")
    plt.title("Initial vs Final Playlist Components")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "components_comparison.png", dpi=200)
    plt.close()

    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"\nFinal score: {result['best_score']:.6f}")
    print(f"Improvement: {result['best_score'] - initial_score:.6f}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print("\nFinal components:")
    print(json.dumps(result["best_detailed"], ensure_ascii=False, indent=2))
    print("\nSaved files:")
    for p in sorted(out.iterdir()):
        print(f"  - {p}")
    print("\nFinal playlist (first 10 tracks):")
    print(final_playlist_df.head(10).to_string(index=False))


# ============================================================================
# Точка входа
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genetic Algorithm for Spotify Playlist Optimization"
    )

    # Основные параметры
    parser.add_argument("--dataset", default="dataset.csv", help="Path to Spotify dataset CSV")
    parser.add_argument(
        "--scenario", default="party",
        choices=list(TARGET_PROFILES.keys()),
        help="Target mood scenario"
    )
    parser.add_argument("--playlist-size", type=int, default=20, help="Number of tracks in playlist")
    parser.add_argument("--candidate-pool-size", type=int, default=800, help="Top N tracks to consider")
    parser.add_argument("--target-duration-min", type=float, default=60, help="Target playlist duration (minutes)")

    # Параметры генетического алгоритма
    parser.add_argument("--population-size", type=int, default=100, help="Population size")
    parser.add_argument("--generations", type=int, default=100, help="Number of generations")
    parser.add_argument("--mutation-rate", type=float, default=0.15, help="Probability of mutation per individual")
    parser.add_argument("--mutation-strength", type=float, default=0.15, help="Fraction of tracks to replace during mutation")
    parser.add_argument("--crossover-rate", type=float, default=0.85, help="Probability of crossover vs cloning")
    parser.add_argument("--elitism-count", type=int, default=5, help="Number of top individuals preserved each generation")
    parser.add_argument("--tournament-size", type=int, default=5, help="Tournament size for selection")

    # Прочее
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output-dir", default="outputs_genetic", help="Output directory")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress bar")

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("GENETIC ALGORITHM FOR PLAYLIST OPTIMIZATION")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Scenario: {args.scenario}")
    print(f"  Playlist size: {args.playlist_size}")
    print(f"  Candidate pool: {args.candidate_pool_size}")
    print(f"  Population size: {args.population_size}")
    print(f"  Generations: {args.generations}")
    print(f"  Mutation rate: {args.mutation_rate}")
    print(f"  Crossover rate: {args.crossover_rate}")
    print(f"  Elitism count: {args.elitism_count}")
    print(f"  Tournament size: {args.tournament_size}")
    print(f"  Seed: {args.seed}")

    start_time = time.time()

    # Создаём задачу (та же, что в LLM-оптимизаторе)
    problem = PlaylistProblem(
        dataset_path=args.dataset,
        scenario=args.scenario,
        playlist_size=args.playlist_size,
        candidate_pool_size=args.candidate_pool_size,
        target_duration_min=args.target_duration_min,
        seed=args.seed,
    )

    print(f"\nDataset loaded: {len(problem.data)} rows")
    print(f"Candidate pool: {len(problem.candidate_pool)} tracks")

    # Создаём и запускаем генетический алгоритм
    ga = GeneticPlaylistOptimizer(
        problem=problem,
        population_size=args.population_size,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        mutation_strength=args.mutation_strength,
        crossover_rate=args.crossover_rate,
        elitism_count=args.elitism_count,
        tournament_size=args.tournament_size,
        seed=args.seed,
    )

    result = ga.run(verbose=not args.quiet)

    elapsed = time.time() - start_time

    # Сохраняем результаты
    save_results(problem, result, ga.history, args.output_dir, elapsed, args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nERROR:")
        print(exc)
        sys.exit(1)