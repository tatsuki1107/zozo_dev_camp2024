from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from obp.policy import BaseOfflinePolicyLearner
from sklearn.utils import check_random_state
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from utils import BatchBanditFeedbackDataset


@dataclass
class RegBasedPolicyLearner(BaseOfflinePolicyLearner):
    dim_context: int = 5
    n_actions: int = 100
    hidden_layer_size: tuple[int] = (5, 5, 5)
    activation: str = "elu"
    batch_size: int = 16
    learning_rate_init: float = 0.01
    gamma: float = 0.98
    alpha: float = 1e-6
    reg: float = 0.0
    solver: str = "adagrad"
    max_iter: int = 30
    random_state: int = 12345

    def __post_init__(self) -> None:
        self.random_ = check_random_state(self.random_state)
        self.train_loss, self.train_value, self.test_value = [], [], []

        self.input_size = self.dim_context

        if self.activation == "elu":
            self.activation_layer = nn.ELU
        else:
            raise ValueError(f"{self.activation} is not implemented.")

        self.layer_list = []
        for i, h in enumerate(self.hidden_layer_size):
            self.layer_list.append((f"l{i}", nn.Linear(self.input_size, h)))
            self.layer_list.append((f"a{i}", self.activation_layer()))
            self.input_size = h

        self.layer_list.append(("output", nn.Linear(self.input_size, self.n_actions)))
        self.nn_model = nn.Sequential(OrderedDict(self.layer_list))

    def _init_scheduler(self) -> tuple[ExponentialLR, optim.Optimizer]:
        if self.solver == "adagrad":
            optimizer = optim.Adagrad(self.nn_model.parameters(), lr=self.learning_rate_init, weight_decay=self.alpha)
        else:
            raise ValueError(f"{self.solver} is not implemented.")

        scheduler = ExponentialLR(optimizer, gamma=self.gamma)

        return scheduler, optimizer

    def _create_train_data_for_opl(
        self, context: np.ndarray, action: np.ndarray, reward: np.ndarray, pscore: np.ndarray
    ) -> DataLoader:
        dataset = BatchBanditFeedbackDataset(
            context=torch.from_numpy(context).float(),
            action=torch.from_numpy(action).long(),
            reward=torch.from_numpy(reward).float(),
            pscore=torch.from_numpy(pscore).float(),
        )
        return DataLoader(dataset, batch_size=self.batch_size)

    def fit(self, dataset_train: dict, dataset_test: dict) -> None:
        context, action, reward = dataset_train["context"], dataset_train["action"], dataset_train["reward"]
        pscore = dataset_train["pscore"]

        training_data_loader = self._create_train_data_for_opl(
            context=context, action=action, reward=reward, pscore=pscore
        )

        scheduler, optimizer = self._init_scheduler()
        q_x_a_train, q_x_a_test = dataset_train["expected_reward"], dataset_test["expected_reward"]

        for _ in range(self.max_iter):
            loss_epoch = 0.0
            self.nn_model.train()
            for context_, action_, reward_, _ in training_data_loader:
                optimizer.zero_grad()
                q_hat = self.nn_model(context_)
                idx = torch.arange(action_.shape[0], dtype=torch.long)
                loss = ((reward_ - q_hat[idx, action_]) ** 2).mean()
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()

            self.train_loss.append(loss_epoch)
            scheduler.step()
            pi_train = self.predict(dataset_train)
            self.train_value.append((q_x_a_train * pi_train).sum(1).mean())
            pi_test = self.predict(dataset_test)
            self.test_value.append((q_x_a_test * pi_test).sum(1).mean())

    def predict(self, dataset: dict) -> np.ndarray:
        self.nn_model.eval()
        context = torch.from_numpy(dataset["context"]).float()
        q_hat = self.nn_model(context).detach().numpy()

        # common regression-based policy, which is "deterministic"
        pi = np.zeros_like(q_hat)
        pi[np.arange(q_hat.shape[0]), np.argmax(q_hat, axis=1)] = 1.0

        return pi


@dataclass
class PolicyLearnerOverActionEmbedSpaces(BaseOfflinePolicyLearner):
    dim_context: int = 5
    n_actions: int = 100
    n_category: int = 10
    hidden_layer_size: tuple[int] = (5, 5, 5)
    activation: str = "elu"
    batch_size: int = 16
    learning_rate_init: float = 0.01
    gamma: float = 0.98
    alpha: float = 1e-6
    reg: float = 0.0
    log_eps: float = 1e-10
    solver: str = "adagrad"
    max_iter: int = 30
    random_state: int = 12345

    def __post_init__(self) -> None:
        self.random_ = check_random_state(self.random_state)
        self.train_loss, self.train_value, self.test_value = [], [], []

        self.input_size = self.dim_context

        if self.activation == "elu":
            self.activation_layer = nn.ELU
        else:
            raise ValueError(f"{self.activation} is not implemented.")

        self.layer_list = []
        for i, h in enumerate(self.hidden_layer_size):
            self.layer_list.append((f"l{i}", nn.Linear(self.input_size, h)))
            self.layer_list.append((f"a{i}", self.activation_layer()))
            self.input_size = h

        self.layer_list.append(("output", nn.Linear(self.input_size, self.n_category)))
        # opl aims to learn a decision-making policy.
        self.layer_list.append(("softmax", nn.Softmax(dim=1)))
        self.nn_model = nn.Sequential(OrderedDict(self.layer_list))

    def _init_scheduler(self) -> tuple[ExponentialLR, optim.Optimizer]:
        if self.solver == "adagrad":
            optimizer = optim.Adagrad(self.nn_model.parameters(), lr=self.learning_rate_init, weight_decay=self.alpha)
        else:
            raise ValueError(f"{self.solver} is not implemented.")

        scheduler = ExponentialLR(optimizer, gamma=self.gamma)

        return scheduler, optimizer

    def _create_train_data_for_opl(
        self, context: np.ndarray, action: np.ndarray, reward: np.ndarray, pscore_e: np.ndarray
    ) -> DataLoader:
        dataset = BatchBanditFeedbackDataset(
            context=torch.from_numpy(context).float(),
            action=torch.from_numpy(action).long(),
            reward=torch.from_numpy(reward).float(),
            pscore=torch.from_numpy(pscore_e).float(),
        )

        return DataLoader(dataset, batch_size=self.batch_size)

    def _marginalization(self, pi_b: np.ndarray, dataset: dict) -> np.ndarray:
        p_e_a = []
        for d in range(dataset["action_embed"].shape[-1]):
            p_e_a.append(dataset["p_e_a"][:, dataset["action_embed"][:, d], d])

        p_e_a = np.array(p_e_a).T.prod(axis=2)
        p_e_x_pi_b = (dataset["pi_b"][:, :, 0] * p_e_a).sum(axis=1)

        return p_e_x_pi_b

    def fit(self, dataset_train: dict, dataset_test: dict) -> None:
        context, category, reward = dataset_train["context"], dataset_train["action_embed"], dataset_train["reward"]
        pscore_e = self._marginalization(dataset_train["pi_b"], dataset_train)

        category = category.reshape(-1)

        training_data_loader = self._create_train_data_for_opl(
            context=context,
            action=category,
            reward=reward,
            pscore_e=pscore_e,
        )

        scheduler, optimizer = self._init_scheduler()
        q_x_a_train, q_x_a_test = dataset_train["expected_reward"], dataset_test["expected_reward"]
        for _ in range(self.max_iter):
            loss_epoch = 0.0
            self.nn_model.train()
            for (
                context_,
                category_,
                reward_,
                pscore_e_,
            ) in training_data_loader:
                optimizer.zero_grad()
                pi_theta = self.nn_model(context_)
                loss = -self._estimate_policy_gradient(
                    action=category_,
                    reward=reward_,
                    pscore=pscore_e_,
                    pi_theta=pi_theta,
                ).mean()
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
            self.train_loss.append(loss_epoch)
            scheduler.step()
            pi_train = self.predict(dataset_train)
            self.train_value.append((q_x_a_train * pi_train).sum(1).mean())
            pi_test = self.predict(dataset_test)
            self.test_value.append((q_x_a_test * pi_test).sum(1).mean())

    def _estimate_policy_gradient(
        self,
        action: torch.Tensor,
        reward: torch.Tensor,
        pscore: torch.Tensor,
        pi_theta: torch.Tensor,
    ) -> torch.Tensor:
        current_pi = pi_theta.detach()
        log_prob = torch.log(pi_theta + self.log_eps)
        idx = torch.arange(action.shape[0], dtype=torch.long)

        iw = current_pi[idx, action] / pscore
        estimated_policy_grad_arr = (iw * reward + self.reg) * log_prob[idx, action]

        return estimated_policy_grad_arr

    def predict(self, dataset: dict) -> np.ndarray:
        self.nn_model.eval()
        context = torch.from_numpy(dataset["context"]).float()
        p_e_x_pi_theta = self.nn_model(context).detach().numpy()

        n_rounds, e_a = dataset["n_rounds"], dataset["action_context"].reshape(-1)
        overall_policy = np.zeros((n_rounds, self.n_actions))

        best_actions_given_x_c = []
        for e in range(self.n_category):
            # Assumption of no direct effect.
            best_action_given_x_c = self.random_.choice(np.where(e_a == e)[0], size=n_rounds)
            best_actions_given_x_c.append(best_action_given_x_c)

        best_actions_given_x_c = np.array(best_actions_given_x_c).T
        overall_policy[np.arange(n_rounds)[:, None], best_actions_given_x_c] = p_e_x_pi_theta

        return overall_policy
