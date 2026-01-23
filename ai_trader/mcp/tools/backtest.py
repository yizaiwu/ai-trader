"""MCP tools for running backtests."""

import asyncio
import time
from pathlib import Path
from typing import List, Optional

import yaml
from fastmcp import Context

from ai_trader.cli import _load_strategy_class
from ai_trader.core.logging import get_logger
from ai_trader.mcp.models import (
    AnalyzerResults,
    BacktestResult as MCPBacktestResult,
    QuickBacktestRequest,
    RunBacktestRequest,
)
from ai_trader.utils.backtest import run_backtest, BacktestResult

logger = get_logger(__name__)


async def run_backtest_tool(
    request: RunBacktestRequest,
    ctx: Context,
) -> MCPBacktestResult:
    """
    Run a backtest from a YAML configuration file.
    """
    try:
        await ctx.info(f"Loading configuration from {request.config_file}")

        # Load config file
        config_path = Path(request.config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {request.config_file}")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Apply overrides
        if request.strategy:
            config["strategy"]["class"] = request.strategy
        if request.cash is not None:
            config["broker"]["cash"] = request.cash
        if request.commission is not None:
            config["broker"]["commission"] = request.commission

        # Load strategy class
        strategy_config = config["strategy"]
        strategy_class = _load_strategy_class(strategy_config["class"])
        await ctx.info(f"Strategy loaded: {strategy_class.__name__}")

        # Get common parameters
        data_config = config.get("data", {})
        if "file" not in data_config:
            raise ValueError("Only single-stock backtests (data.file) are supported via MCP")

        await ctx.info(f"Data file: {data_config['file']}")
        await ctx.info("Starting backtest execution...")

        # Run backtest in executor
        start_time = time.time()
        result_obj = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: run_backtest(
                strategy_class=strategy_class,
                data_source=data_config["file"],
                cash=config["broker"]["cash"],
                commission=config["broker"]["commission"],
                start_date=data_config.get("start_date"),
                end_date=data_config.get("end_date"),
                strategy_params=strategy_config.get("params", {}),
            ),
        )
        execution_time = time.time() - start_time

        # Format results for MCP
        result = _format_mcp_results(result_obj, execution_time)
        await ctx.info(f"Backtest complete: {result.return_pct:.2f}% return")

        return result

    except Exception as e:
        error_msg = f"Backtest failed: {str(e)}"
        await ctx.error(error_msg)
        logger.exception("Error in run_backtest_tool")
        raise


async def quick_backtest_tool(
    request: QuickBacktestRequest,
    ctx: Context,
) -> MCPBacktestResult:
    """
    Quick backtest without configuration file.
    """
    try:
        await ctx.info(f"Strategy: {request.strategy_name}")
        await ctx.info(f"Data file: {request.data_file}")

        # Load strategy class
        strategy_class = _load_strategy_class(request.strategy_name)
        await ctx.info(f"Strategy loaded: {strategy_class.__name__}")

        await ctx.info("Starting backtest execution...")

        # Run backtest in executor
        start_time = time.time()
        result_obj = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: run_backtest(
                strategy_class=strategy_class,
                data_source=request.data_file,
                cash=request.cash,
                commission=request.commission,
                start_date=request.start_date,
                end_date=request.end_date,
            ),
        )
        execution_time = time.time() - start_time

        # Format results for MCP
        result = _format_mcp_results(result_obj, execution_time)
        await ctx.info(f"Backtest complete: {result.return_pct:.2f}% return")

        return result

    except Exception as e:
        error_msg = f"Quick backtest failed: {str(e)}"
        await ctx.error(error_msg)
        logger.exception("Error in quick_backtest_tool")
        raise


def _format_mcp_results(
    result: BacktestResult,
    execution_time: float,
) -> MCPBacktestResult:
    """
    Convert internal BacktestResult to MCPBacktestResult.
    """
    analyzer_results = AnalyzerResults()
    
    # Calculate some basic metrics from trades
    total_trades = len(result.trades) // 2  # Close approximation if every buy has a sell
    won_trades = 0
    # In this simple engine, we can calculate real PnL from trades list if needed
    # For now, let's just populate the main ones
    
    return MCPBacktestResult(
        strategy_name=result.strategy_name,
        initial_value=result.initial_value,
        final_value=result.final_value,
        profit_loss=result.profit_loss,
        return_pct=result.return_pct,
        analyzers=analyzer_results,
        execution_time_seconds=execution_time,
    )
