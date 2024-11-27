import logging
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
from obp.dataset import SyntheticBanditDatasetWithActionEmbeds
from obp.dataset import linear_behavior_policy
from obp.dataset import logistic_reward_function
import pandas as pd
from tqdm import tqdm

from utils import PolicyLearnerOverActionEmbedSpaces
from utils import RegBasedPolicyLearner
from utils import visualize_learning_curve
from utils import visualize_test_value


logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg):
    result_path = Path(HydraConfig.get().run.dir)
    result_path.mkdir(parents=True, exist_ok=True)

    curve_path = result_path / "learning_curve"
    curve_path.mkdir(parents=True, exist_ok=True)

    dataset = SyntheticBanditDatasetWithActionEmbeds(
        n_actions=cfg.n_actions,
        dim_context=cfg.dim_context,
        reward_type=cfg.reward_type,
        reward_function=logistic_reward_function,
        behavior_policy_function=linear_behavior_policy,
        beta=cfg.beta,
        n_cat_dim=cfg.n_cat_dim,
        n_cat_per_dim=cfg.n_cat_per_dim,
        p_e_a_param_std=8,  # nealy-deterministic
        random_state=cfg.random_state,
    )
    test_data = dataset.obtain_batch_bandit_feedback(n_rounds=cfg.n_test_samplesizes)
    test_data["unique_action_context"] = dataset.action_context_reg

    is_popular_actions = np.zeros(cfg.n_actions)
    is_popular_actions[np.unique(test_data["action"], return_counts=True)[1].argsort()[::-1][: cfg.n_popular_items]] = 1
    test_data["is_popular_action"] = is_popular_actions

    pi_b_value = (test_data["pi_b"][:, :, 0] * test_data["expected_reward"]).sum(1).mean()
    logger.info(f"pi_b_value: {pi_b_value}")

    result_df_list = []
    for n_val_size in cfg.n_val_samplesizes:
        curve_df = pd.DataFrame()
        test_policy_value_list, test_policy_popular_prob_list = [], []
        tqdm_ = tqdm(range(cfg.n_val_seeds), desc=f"sample size: {n_val_size}")
        for _ in tqdm_:
            logged_data = dataset.obtain_batch_bandit_feedback(n_rounds=n_val_size)

            true_value_of_learned_policies = dict()
            popular_probs_of_learned_policies = dict()

            reg = RegBasedPolicyLearner(
                n_actions=cfg.n_actions,
                dim_context=cfg.dim_context,
                max_iter=cfg.max_iter,
            )
            reg.fit(logged_data, test_data)
            reg_pi = reg.predict(logged_data)
            true_value_of_learned_policies["reg"] = (reg_pi * logged_data["expected_reward"]).sum(1).mean()
            popular_probs_of_learned_policies["reg"] = (reg_pi * test_data["is_popular_action"]).sum(1).mean()

            reg_ = pd.DataFrame([reg.test_value, ["reg"] * cfg.max_iter], index=["value", "method"]).T.reset_index()

            mips_pg = PolicyLearnerOverActionEmbedSpaces(
                n_actions=cfg.n_actions,
                dim_context=cfg.dim_context,
                n_category=cfg.n_cat_per_dim,
                max_iter=cfg.max_iter,
            )

            mips_pg.fit(logged_data, test_data)
            mips_pi = mips_pg.predict(logged_data)
            true_value_of_learned_policies["mips"] = (mips_pi * logged_data["expected_reward"]).sum(1).mean()
            popular_probs_of_learned_policies["mips"] = (mips_pi * test_data["is_popular_action"]).sum(1).mean()

            test_policy_value_list.append(true_value_of_learned_policies)
            test_policy_popular_prob_list.append(popular_probs_of_learned_policies)

            mips_ = pd.DataFrame(
                [mips_pg.test_value, ["mips"] * cfg.max_iter], index=["value", "method"]
            ).T.reset_index()

            curve_df = pd.concat([curve_df, pd.concat([reg_, mips_])])

        logger.info(tqdm_)
        curve_df.reset_index(inplace=True)
        curve_df["rel_value"] = curve_df["value"] / pi_b_value
        visualize_learning_curve(curve_df=curve_df, img_path=curve_path / f"n_val_size_{n_val_size}.png")

        policy_value_result_df = (
            pd.DataFrame(test_policy_value_list)
            .stack()
            .reset_index(1)
            .rename(columns={"level_1": "method", 0: "policy_value"})
            .reset_index(drop=True)
        )
        policy_value_result_df["num_data"] = n_val_size
        policy_value_result_df["rel_value"] = policy_value_result_df["policy_value"] / pi_b_value

        policy_popular_prob_result_df = (
            pd.DataFrame(test_policy_popular_prob_list)
            .stack()
            .reset_index(1)
            .rename(columns={"level_1": "method", 0: "popular_prob"})
            .reset_index(drop=True)
        )

        result_df_list.append(
            pd.concat([policy_value_result_df, policy_popular_prob_result_df[["popular_prob"]]], axis=1)
        )

    result_df = pd.concat(result_df_list).reset_index(level=0)
    visualize_test_value(result_df=result_df, img_path=result_path)


if __name__ == "__main__":
    main()
