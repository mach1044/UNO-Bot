import torch
import random
from collections import deque
from pathlib import Path

from LearningBot import LearningBot
from GameEngine import GameEngine

DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parent
    / "checkpoints"
    / "learning_bot_from_scratch.pt"
)

class train_learning_bot:
    def __init__(self, epsilon: float):
        self.bot = LearningBot(epsilon)
        self.replay_memory = deque(maxlen=30_000)
        self.games_trained = 0

        self.batch_size = 64

        self.loss_function = (
            torch.nn.BCEWithLogitsLoss()
        )

        self.optimizer = torch.optim.Adam(
            self.bot.model.parameters(),
            lr=0.0003
        )

    def load_checkpoint(self):
        checkpoint = torch.load(
            DEFAULT_CHECKPOINT,
            map_location="cpu",
            weights_only=True,
        )

        self.bot.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        self.games_trained = checkpoint["games_trained"]
        print(
            f"Loaded checkpoint after "                                                                                                               
            f"{self.games_trained} games"
        )

    def save_checkpoint(self):
        DEFAULT_CHECKPOINT.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            {
                "model_state_dict": self.bot.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "games_trained": self.games_trained,
                "state_size": 80,
                "move_size": 77,
            },
            DEFAULT_CHECKPOINT,
        )

        print(f"Saved checkpoint to {DEFAULT_CHECKPOINT}")

    def run_game(self):
        engine = GameEngine()
        self.bot.episode_decisions.clear()

        try:
            winner = engine.game(
                self.bot,
                self.bot,
                max_turns=1000,
            )
        except RuntimeError as error:
            if str(error) != "No cards are available to draw":
                raise

            print(
                f"Game {self.games_trained + 1} discarded: "
                "no cards are available to draw",
                flush=True,
            )
            self.bot.episode_decisions.clear()
            return

        if winner is None:
            self.bot.episode_decisions.clear()
            return None

        if winner is engine.player1:
            winner_id = 0
        else:
            winner_id = 1

        for state, move, player_id in self.bot.episode_decisions:
            if player_id == winner_id:
                target = 1.0
            else:
                target = 0.0

            self.replay_memory.append(
                (state, move, target)
            )

        self.bot.episode_decisions.clear()

    def train_batch(self):
        if len(self.replay_memory) < self.batch_size:
            return None

        batch = random.sample(
            list(self.replay_memory),
            self.batch_size,
        )

        states = torch.stack([
            experience[0] for experience in batch
        ])

        moves = torch.stack([
            experience[1] for experience in batch
        ])

        targets = torch.tensor(
            [experience[2] for experience in batch],
            dtype=torch.float32,
        )

        self.bot.model.train()

        predictions = self.bot.model(
            states,
            moves,
        )

        loss = self.loss_function(
            predictions,
            targets,
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, number_of_games: int):
        for game_number in range(1, number_of_games + 1):
            self.run_game()
            loss = self.train_batch()
            self.games_trained += 1

            if game_number % 100 == 0:
                if loss is None:
                    loss_text = "waiting for replay data"
                else:
                    loss_text = f"{loss:.4f}"

                print(
                    f"{game_number} games completed | "
                    f"loss {loss_text} | "
                    f"replay {len(self.replay_memory)}"
                )
                self.save_checkpoint()

        self.save_checkpoint()

if __name__ == "__main__":
    epsilon = 0.15
    trainer = train_learning_bot(epsilon)

    if DEFAULT_CHECKPOINT.exists():
        trainer.load_checkpoint()

    trainer.train(50000)
