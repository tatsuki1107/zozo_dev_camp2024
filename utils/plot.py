from pathlib import PosixPath

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def visualize_learning_curve(curve_df: pd.DataFrame, img_path: PosixPath) -> None:
    plt.style.use("ggplot")

    fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)
    sns.lineplot(
        linewidth=5,
        x="index",
        y="rel_value",
        hue="method",
        style="method",
        ax=ax,
        data=curve_df,
    )
    ax.set_title("Learning curve", fontsize=20)
    ax.set_ylabel("Rerative policy value", fontsize=30)
    ax.set_xlabel("epochs", fontsize=25)
    ax.tick_params(axis="y", labelsize=20)

    ax.legend(fontsize=15, title="method", title_fontsize=15, loc="lower right")
    plt.show()
    plt.savefig(img_path)
    plt.close()


def visualize_test_value(result_df: pd.DataFrame, img_path: PosixPath) -> None:
    x = result_df["num_data"].unique().astype(int)

    plt.style.use("ggplot")
    for y in ["rel_value", "popular_prob"]:
        fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)

        sns.lineplot(
            linewidth=5,
            markersize=15,
            markers=True,
            x="num_data",
            y=y,
            hue="method",
            style="method",
            ax=ax,
            data=result_df,
            ci=None,
            legend=False,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(x, fontsize=20)
        ax.set_xlabel("")

        ax.set_ylabel("")
        ax.tick_params(axis="y", labelsize=20)

        plt.show()
        plt.savefig(img_path / f"{y}.png")
        plt.close()
