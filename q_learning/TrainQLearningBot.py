import torch
import random
from collections import deque
from pathlib import Path

from q_learning.QLearningBot import (
    QLearningBot,
    QNetwork,
)

from engine.GameEngine import GameEngine

DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parent
    / "checkpoints"
    / "q_learning_bot.pt"
)

class TrainQLearningBot:
    def __init__(self, epsilon: float):
        self.bot = QLearningBot(epsilon)

        self.target_model = QNetwork()
        self.target_model.load_state_dict(
            self.bot.model.state_dict()
        )
        self.target_model.eval()

        self.replay_memory = deque(maxlen=50_000)

        self.games_trained = 0
        self.training_steps = 0

        self.batch_size = 64
        self.gamma = 0.99
        self.target_update_steps = 1000

        self.loss_function = (
            torch.nn.SmoothL1Loss()
        )

        self.optimizer = torch.optim.Adam(
            self.bot.model.parameters(),
            lr=0.0003
        )

    def update_target_model(self):
        self.target_model.load_state_dict(
            self.bot.model.state_dict()
        )
        self.target_model.eval()

    def load_checkpoint(self):
        checkpoint = torch.load(
            DEFAULT_CHECKPOINT,
            map_location="cpu",
            weights_only=True,
        )

        self.bot.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        target_state = checkpoint.get(
            "target_model_state_dict",
            checkpoint["model_state_dict"],
        )

        self.target_model.load_state_dict(
            target_state
        )
        self.target_model.eval()

        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        self.games_trained = checkpoint["games_trained"]

        self.training_steps = checkpoint.get(
            "training_steps",
            0,
        )

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
                "target_model_state_dict": self.target_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "games_trained": self.games_trained,
                "state_size": 80,
                "move_size": 77,
                "training_steps": self.training_steps
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

        previous_decisions = {
            0: None,
            1: None,
        }

        for state, action, player_id, legal_actions in self.bot.episode_decisions:
            previous = previous_decisions[player_id]

            if previous is not None:
                previous_state, previous_action = previous

                self.replay_memory.append(
                    (
                        previous_state,
                        previous_action,
                        0.0,
                        state,
                        legal_actions,
                        False,
                    )
                )

            previous_decisions[player_id] = (
                state,
                action,
            )

        for player_id, previous in previous_decisions.items():
            if previous is None:
                continue

            state, action = previous

            if player_id == winner_id:
                reward = 1.0
            else:
                reward = 0.0

            self.replay_memory.append(
                (
                    state,
                    action,
                    reward,
                    None,
                    None,
                    True,
                )
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
            transition[0] for transition in batch
        ])

        actions = torch.stack([
            transition[1] for transition in batch
        ])

        rewards = torch.tensor(
            [transition[2] for transition in batch],
            dtype=torch.float32,
        )

        dones = torch.tensor(
            [transition[5] for transition in batch],
            dtype=torch.bool,
        )

        targets = rewards.clone()

        with torch.no_grad():
            self.bot.model.eval()

            for index, transition in enumerate(batch):
                if dones[index]:
                    continue

                next_state = transition[3]
                next_actions = transition[4]

                repeated_next_states = (
                    next_state.unsqueeze(0).expand(
                        len(next_actions),
                        -1,
                    )
                )

                online_scores = self.bot.model(
                    repeated_next_states,
                    next_actions,
                )

                best_action_index = int(
                    torch.argmax(online_scores).item()
                )

                target_scores = self.target_model(
                    repeated_next_states,
                    next_actions,
                )

                best_next_q = target_scores[
                    best_action_index
                ]

                targets[index] += (
                    self.gamma * best_next_q
                )

        self.bot.model.train()

        predictions = self.bot.model(
            states,
            actions,
        )

        loss = self.loss_function(
            predictions,
            targets,
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.training_steps += 1

        if (
            self.training_steps
            % self.target_update_steps
            == 0
        ):
            self.update_target_model()

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
    epsilon = 0.10
    trainer = TrainQLearningBot(epsilon)

    if DEFAULT_CHECKPOINT.exists():
        trainer.load_checkpoint()

    trainer.train(50000)
