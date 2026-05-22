"""
Transaction Cost Analysis Module.

Analyzes portfolio performance under different transaction cost structures
and rebalancing frequencies.

"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parent.parent  # → code/


class TransactionCostAnalyzer:
    """Analyse the impact of transaction costs on portfolio performance."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        if config_path is None:
            config_path = _ROOT / "config" / "config.yaml"

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.cost_structures: Dict[str, float] = self.config["transaction_costs"][
            "cost_structures"
        ]
        self.rebalance_frequencies: Dict[str, int] = self.config["transaction_costs"][
            "rebalance_frequencies"
        ]
        self.output_dir = Path(self.config["transaction_costs"]["analysis_output"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Core analysis                                                         #
    # ------------------------------------------------------------------ #

    def analyze_strategy_with_costs(
        self,
        strategy_name: str,
        portfolio_values_no_cost: List[float],
        portfolio_weights_history: List[np.ndarray],
        dates: List,
        cost_structure: str = "standard",
        rebalance_freq: int = 20,
    ) -> Dict:
        """
        Re-simulate a strategy with explicit transaction costs applied at
        each rebalancing event and return a performance comparison.

        Parameters
        ----------
        strategy_name : str
        portfolio_values_no_cost : list[float]
            Baseline portfolio values without transaction costs.
        portfolio_weights_history : list[np.ndarray]
            Weight vector at each time step.
        dates : list
            Corresponding date labels.
        cost_structure : str
            Key into config['transaction_costs']['cost_structures'].
        rebalance_freq : int
            Number of days between rebalances.

        Returns
        -------
        dict
            Performance metrics with and without costs.
        """
        cost_pct = self.cost_structures[cost_structure]
        values_with = [float(portfolio_values_no_cost[0])]
        total_costs = 0.0

        for i in range(1, len(portfolio_values_no_cost)):
            base_val = portfolio_values_no_cost[i]

            if i % rebalance_freq == 0 and i > 0 and i < len(portfolio_weights_history):
                w_prev = portfolio_weights_history[i - 1]
                w_curr = portfolio_weights_history[i]
                turnover = np.abs(w_curr - w_prev).sum()
                cost = turnover * portfolio_values_no_cost[i - 1] * cost_pct
                total_costs += cost
                adjusted_val = base_val - cost
            else:
                adjusted_val = base_val

            values_with.append(float(adjusted_val))

        ret_no_cost = np.diff(portfolio_values_no_cost) / np.array(
            portfolio_values_no_cost[:-1]
        )
        ret_with_cost = np.diff(values_with) / np.array(values_with[:-1])

        return {
            "strategy": strategy_name,
            "cost_structure": cost_structure,
            "rebalance_freq": rebalance_freq,
            "total_transaction_costs": round(total_costs, 2),
            "final_value_no_cost": round(portfolio_values_no_cost[-1], 2),
            "final_value_with_cost": round(values_with[-1], 2),
            "cost_impact": round(portfolio_values_no_cost[-1] - values_with[-1], 2),
            "cost_impact_pct": round(
                (portfolio_values_no_cost[-1] - values_with[-1])
                / portfolio_values_no_cost[-1]
                * 100,
                4,
            ),
            "sharpe_no_cost": self._sharpe(ret_no_cost),
            "sharpe_with_cost": self._sharpe(ret_with_cost),
            "max_drawdown_no_cost": self._max_drawdown(portfolio_values_no_cost),
            "max_drawdown_with_cost": self._max_drawdown(values_with),
        }

    def analyze_rebalancing_frequency(
        self,
        strategy_name: str,
        portfolio_values_base: List[float],
        portfolio_weights_history: List[np.ndarray],
        dates: List,
    ) -> pd.DataFrame:
        """Grid-search over all (rebalance_freq × cost_structure) combinations."""
        rows = []
        for freq_name, freq_days in self.rebalance_frequencies.items():
            for cost_name in self.cost_structures:
                m = self.analyze_strategy_with_costs(
                    strategy_name,
                    portfolio_values_base,
                    portfolio_weights_history,
                    dates,
                    cost_structure=cost_name,
                    rebalance_freq=freq_days,
                )
                m["frequency_name"] = freq_name
                rows.append(m)
        return pd.DataFrame(rows)

    def compare_with_without_costs(
        self,
        strategies: Dict[str, Dict],
        output_path: str | None = None,
    ) -> pd.DataFrame:
        rows = []
        for name, d in strategies.items():
            rows.append(
                {
                    "Strategy": name,
                    "Scenario": "No Costs",
                    "Final Value ($)": d.get("final_value_no_cost", 0),
                    "Annual Return (%)": d.get("annual_return_no_cost", 0),
                    "Sharpe Ratio": d.get("sharpe_no_cost", 0),
                    "Max Drawdown (%)": d.get("max_drawdown_no_cost", 0),
                }
            )
            rows.append(
                {
                    "Strategy": name,
                    "Scenario": "With Costs (0.1%)",
                    "Final Value ($)": d.get("final_value_with_cost", 0),
                    "Annual Return (%)": d.get("annual_return_with_cost", 0),
                    "Sharpe Ratio": d.get("sharpe_with_cost", 0),
                    "Max Drawdown (%)": d.get("max_drawdown_with_cost", 0),
                }
            )

        df = pd.DataFrame(rows)
        if output_path:
            df.to_csv(output_path, index=False)
        return df

    # ------------------------------------------------------------------ #
    # Plots                                                                 #
    # ------------------------------------------------------------------ #

    def plot_cost_impact(
        self,
        results_df: pd.DataFrame,
        save_path: str | None = None,
    ) -> plt.Figure:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        pivot_impact = results_df.pivot_table(
            values="cost_impact_pct",
            index="frequency_name",
            columns="cost_structure",
            aggfunc="mean",
        )
        pivot_impact.plot(kind="bar", ax=axes[0, 0])
        axes[0, 0].set(
            title="Cost Impact (%) by Rebalancing Frequency",
            xlabel="Rebalancing Frequency",
            ylabel="Cost Impact (%)",
        )
        axes[0, 0].legend(title="Cost Structure")
        axes[0, 0].grid(True, alpha=0.3)

        sharpe_data = results_df.groupby("frequency_name")[
            ["sharpe_no_cost", "sharpe_with_cost"]
        ].mean()
        sharpe_data.plot(kind="bar", ax=axes[0, 1])
        axes[0, 1].set(
            title="Sharpe Ratio: With vs Without Costs",
            xlabel="Rebalancing Frequency",
            ylabel="Sharpe Ratio",
        )
        axes[0, 1].legend(["No Costs", "With Costs"])
        axes[0, 1].grid(True, alpha=0.3)

        pivot_total = results_df.pivot_table(
            values="total_transaction_costs",
            index="frequency_name",
            columns="cost_structure",
            aggfunc="mean",
        )
        pivot_total.plot(kind="bar", ax=axes[1, 0])
        axes[1, 0].set(
            title="Total Transaction Costs by Frequency",
            xlabel="Rebalancing Frequency",
            ylabel="Total Costs ($)",
        )
        axes[1, 0].legend(title="Cost Structure")
        axes[1, 0].grid(True, alpha=0.3)

        optimal = results_df.loc[
            results_df.groupby("cost_structure")["sharpe_with_cost"].idxmax()
        ]
        axes[1, 1].barh(optimal["cost_structure"], optimal["sharpe_with_cost"])
        axes[1, 1].set(
            title="Optimal Sharpe by Cost Structure",
            xlabel="Sharpe Ratio",
            ylabel="Cost Structure",
        )
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------ #
    # Report                                                                #
    # ------------------------------------------------------------------ #

    def generate_cost_report(
        self,
        results_df: pd.DataFrame,
        output_path: str | None = None,
    ) -> str:
        lines = [
            "=" * 80,
            "TRANSACTION COST ANALYSIS REPORT",
            "=" * 80,
            "",
            "SUMMARY STATISTICS",
            "-" * 80,
            f"Average Cost Impact : {results_df['cost_impact_pct'].mean():.2f}%",
            f"Max Cost Impact     : {results_df['cost_impact_pct'].max():.2f}%",
            f"Min Cost Impact     : {results_df['cost_impact_pct'].min():.2f}%",
            "",
            "OPTIMAL REBALANCING FREQUENCY",
            "-" * 80,
        ]
        for cs in results_df["cost_structure"].unique():
            sub = results_df[results_df["cost_structure"] == cs]
            best = sub.loc[sub["sharpe_with_cost"].idxmax()]
            lines += [
                f"\n  Cost Structure : {cs}",
                f"  Optimal Freq   : {best['frequency_name']}",
                f"  Sharpe Ratio   : {best['sharpe_with_cost']:.3f}",
                f"  Cost Impact    : {best['cost_impact_pct']:.2f}%",
            ]

        lines += ["", "COST STRUCTURE COMPARISON", "-" * 80]
        comp = results_df.groupby("cost_structure").agg(
            cost_impact_pct=("cost_impact_pct", "mean"),
            sharpe_with_cost=("sharpe_with_cost", "mean"),
            total_transaction_costs=("total_transaction_costs", "mean"),
        )
        lines.append(comp.to_string())

        report = "\n".join(lines)
        if output_path:
            Path(output_path).write_text(report)
        return report

    # ------------------------------------------------------------------ #
    # Private helpers                                                       #
    # ------------------------------------------------------------------ #

    def _sharpe(self, returns: np.ndarray, rf: float = 0.045) -> float:
        if len(returns) == 0:
            return 0.0
        ann_ret = float(np.mean(returns) * 252)
        ann_vol = float(np.std(returns) * np.sqrt(252))
        return (ann_ret - rf) / ann_vol if ann_vol > 0 else 0.0

    def _max_drawdown(self, values: List[float]) -> float:
        arr = np.array(values, dtype=float)
        peak = np.maximum.accumulate(arr)
        dd = (peak - arr) / np.where(peak > 0, peak, 1)
        return float(-np.max(dd) * 100)


if __name__ == "__main__":
    print("Transaction Cost Analyzer module loaded successfully")
