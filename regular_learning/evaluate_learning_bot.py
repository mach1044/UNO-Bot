from datetime import datetime
import torch
from LearningBot import LearningBot
from engine.GameEngine import GameEngine

from benchmark_bots.HeuristicBot import HeuristicBot
from benchmark_bots.RandomBot import RandomBot
from benchmark_bots.StrongerHeuristicBot import StrongerHeuristicBot

from pathlib import Path

DEFAULT_CHECKPOINT = (
        Path(__file__).resolve().parent
        / "checkpoints"
        / "learning_bot_from_scratch.pt"
)

RESULTS_FILE = (
        Path(__file__).resolve().parent
        / "results"
        / "evaluation_results.txt"
)

class evaluate_learning_bot:
    def run_simulation(self, games: int, opponent: str):
        learning_bot, checkpoint = self.load_learning_bot()
        if opponent == "heuristic":
            enemy_bot = HeuristicBot()
        elif opponent == "stronger_heuristic":
            enemy_bot = StrongerHeuristicBot()
        elif opponent == "random":
            enemy_bot = RandomBot()
        elif opponent == "strong_heuristic":
            enemy_bot = StrongerHeuristicBot()

        simulator = GameEngine()
        wins = 0
        completed_games = 0
        truncated_games = 0
        while completed_games < games:
            simulator.restart_game()

            if completed_games % 2 == 0:
                result = simulator.game(learning_bot, enemy_bot, max_turns=1000)
                learning_player = simulator.player1
            else:
                result = simulator.game(enemy_bot, learning_bot, max_turns=1000)
                learning_player = simulator.player2

            learning_bot.episode_decisions.clear()

            if result == None:
                truncated_games += 1
                continue
            elif result is learning_player:
                wins += 1
            completed_games += 1

        self.write_results(
            completed_games=completed_games,
            wins=wins,
            truncated_games=truncated_games,
            opponent=opponent,
            checkpoint=checkpoint,
        )

    def load_learning_bot(self):
        if not DEFAULT_CHECKPOINT.exists():
            raise FileNotFoundError(
                f"training bot has not been created/found"
            )

        checkpoint = torch.load(
            DEFAULT_CHECKPOINT,
            map_location=torch.device("cpu"),
            weights_only=True,
        )

        learning_bot = LearningBot()

        learning_bot.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        learning_bot.model.eval()

        return learning_bot, checkpoint

    def write_results(
            self,
            completed_games: int,
            wins: int,
            truncated_games: int,
            opponent: str,
            checkpoint: dict,
    ):
        losses = completed_games - wins

        if completed_games == 0:
            win_rate = 0.0
        else:
            win_rate = wins / completed_games * 100

        result = (
            f"Opponent: {opponent}\n"
            f"Checkpoint games trained: "
            f"{checkpoint.get('games_trained', 'unknown')}\n"
            f"Completed games: {completed_games}\n"
            f"Wins: {wins}\n"
            f"Losses: {losses}\n"
            f"Truncated attempts: {truncated_games}\n"
            f"Win rate: {win_rate:.1f}%\n"
            f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{'-' * 40}\n"
        )

        RESULTS_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(result)

        with RESULTS_FILE.open(
            "a",
            encoding="utf-8",
        ) as results_file:
            results_file.write(result)


if __name__ == "__main__":
    evaluator = evaluate_learning_bot()
    evaluator.run_simulation(
        games=5000,
        opponent="stronger_heuristic",
    )
