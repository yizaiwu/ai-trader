"""Command-line interface for ai-trader."""

import importlib
import sys
from pathlib import Path
from typing import Optional

import click
import yaml

from ai_trader.core.logging import get_logger
from ai_trader.utils.backtest import run_backtest

logger = get_logger(__name__)


@click.group()
@click.version_option(version="0.2.0")
def cli():
    """AI Trader - Backtesting framework for algorithmic trading strategies."""
    pass


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--strategy", help="Override strategy from config")
@click.option("--cash", type=float, help="Override initial cash")
@click.option("--commission", type=float, help="Override commission rate")
def run(
    config_file: str,
    strategy: Optional[str],
    cash: Optional[float],
    commission: Optional[float],
):
    """
    Run a backtest from a YAML configuration file.
    """
    # Load config
    with open(config_file) as f:
        config = yaml.safe_load(f)

    # Apply overrides
    if strategy:
        config["strategy"]["class"] = strategy
    if cash is not None:
        config["broker"]["cash"] = cash
    if commission is not None:
        config["broker"]["commission"] = commission

    # Load strategy class
    strategy_config = config["strategy"]
    strategy_class = _load_strategy_class(strategy_config["class"])

    # Get common parameters
    data_config = config.get("data", {})
    
    # Check if single stock or portfolio
    if "file" in data_config:
        # Single stock - use run_backtest()
        click.echo(f"\nRunning backtest: {strategy_class.__name__}")
        click.echo(f"Data: {data_config['file']}\n")

        result = run_backtest(
            strategy_class=strategy_class,
            data_source=data_config["file"],
            cash=config["broker"]["cash"],
            commission=config["broker"]["commission"],
            start_date=data_config.get("start_date"),
            end_date=data_config.get("end_date"),
            strategy_params=strategy_config.get("params", {}),
        )
        click.echo(f"Final Value: ${result.final_value:,.2f} ({result.return_pct:+.2f}%)")

    elif "directory" in data_config:
        click.echo("Error: Portfolio backtests are currently disabled after backtrader removal.", err=True)
        sys.exit(1)
    else:
        click.echo("Error: Config must specify 'data.file'", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--type",
    "strategy_type",
    type=click.Choice(["classic", "portfolio", "all"]),
    default="all",
    help="Type of strategies to list",
)
def list_strategies(strategy_type: str):
    """
    List available trading strategies.
    """
    from ai_trader.backtesting.strategies import classic, portfolio

    click.echo("\n" + "=" * 60)
    click.echo("AVAILABLE STRATEGIES")
    click.echo("=" * 60 + "\n")

    if strategy_type in ("classic", "all"):
        click.echo("Classic Strategies (single stock):")
        click.echo("-" * 40)
        classic_strategies = _get_strategies_from_module(classic)
        for name, cls in classic_strategies:
            doc = cls.__doc__.split("\n")[0] if cls.__doc__ else "No description"
            click.echo(f"  • {name:30s} - {doc}")
        click.echo()

    if strategy_type in ("portfolio", "all"):
        click.echo("Portfolio Strategies (multi-stock):")
        click.echo("-" * 40)
        portfolio_strategies = _get_strategies_from_module(portfolio)
        for name, cls in portfolio_strategies:
            doc = cls.__doc__.split("\n")[0] if cls.__doc__ else "No description"
            click.echo(f"  • {name:30s} - {doc}")
        click.echo()


@cli.command()
@click.argument("symbols", nargs=-1, required=False)
@click.option(
    "--symbols-file", type=click.Path(exists=True), help="File containing symbols (one per line)"
)
@click.option(
    "--market",
    type=click.Choice(["us_stock", "tw_stock", "crypto", "forex", "vix"]),
    default="us_stock",
)
@click.option("--start-date", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end-date", help="End date (YYYY-MM-DD), defaults to today")
@click.option("--output-dir", help="Output directory", default="./data")
def fetch(
    symbols: tuple[str],
    symbols_file: Optional[str],
    market: str,
    start_date: str,
    end_date: Optional[str],
    output_dir: str,
):
    """
    Fetch market data and save to CSV.
    """
    from ai_trader.data.fetchers import (
        CryptoDataFetcher,
        ForexDataFetcher,
        TWStockFetcher,
        USStockFetcher,
        VIXDataFetcher,
    )
    from ai_trader.data.storage import FileManager

    fetcher_factory = {
        "us_stock": (USStockFetcher, "symbol"),
        "tw_stock": (TWStockFetcher, "symbol"),
        "crypto": (CryptoDataFetcher, "ticker"),
        "forex": (ForexDataFetcher, "symbol"),
        "vix": (VIXDataFetcher, None),
    }

    symbol_list = []

    if symbols:
        for sym in symbols:
            symbol_list.extend(s.strip() for s in sym.split(",") if s.strip())

    if symbols_file:
        if symbols:
            click.echo("✗ Error: Cannot specify both symbol arguments and --symbols-file", err=True)
            sys.exit(1)
        with open(symbols_file) as f:
            symbol_list.extend(line.strip() for line in f if line.strip())

    if not symbol_list:
        click.echo("✗ Error: No symbols provided.", err=True)
        sys.exit(1)

    seen = set()
    symbol_list = [s for s in symbol_list if not (s in seen or seen.add(s))]

    click.echo(f"\nFetching {market.upper()} market data for {len(symbol_list)} symbol(s)...")
    market_dir = f"{output_dir}/{market}"

    try:
        if len(symbol_list) == 1 or market in ("forex", "vix"):
            symbol = symbol_list[0]
            if market not in fetcher_factory:
                click.echo(f"✗ Invalid market: {market}", err=True)
                sys.exit(1)

            fetcher_class, symbol_param = fetcher_factory[market]
            fetcher_params = {"start_date": start_date, "end_date": end_date}
            if symbol_param:
                fetcher_params[symbol_param] = symbol

            fetcher = fetcher_class(**fetcher_params)
            df = fetcher.fetch()

            if df is None or df.empty:
                click.echo("✗ No data returned", err=True)
                sys.exit(1)

            file_manager = FileManager(base_data_dir=market_dir)
            actual_end_date = end_date or df.index[-1].strftime("%Y-%m-%d")
            filepath = file_manager.save_to_csv(
                df=df, ticker=symbol, start_date=start_date, end_date=actual_end_date, overwrite=True
            )
            click.echo(f"✓ Data saved to {filepath}")

        else:
            fetcher_class, symbol_param = fetcher_factory[market]
            fetcher_params = {symbol_param: "", "start_date": start_date, "end_date": end_date}
            fetcher = fetcher_class(**fetcher_params)
            successful_data, failed_symbols = fetcher.fetch_batch(symbol_list)

            file_manager = FileManager(base_data_dir=market_dir)
            saved_count = 0
            for symbol, df in successful_data.items():
                actual_end_date = end_date or df.index[-1].strftime("%Y-%m-%d")
                file_manager.save_to_csv(
                    df=df, ticker=symbol, start_date=start_date, end_date=actual_end_date, overwrite=True
                )
                saved_count += 1

            click.echo(f"\n✓ Successfully downloaded {saved_count}/{len(symbol_list)} symbols.")
            if failed_symbols:
                click.echo(f"✗ Failed: {', '.join(failed_symbols)}")

    except Exception as e:
        click.echo(f"\n✗ Failed to fetch data: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("strategy_name")
@click.argument("data_file")
@click.option("--cash", type=float, default=1000000, help="Initial cash")
@click.option("--commission", type=float, default=0.001425, help="Commission rate")
@click.option("--start-date", help="Start date (YYYY-MM-DD)")
@click.option("--end-date", help="End date (YYYY-MM-DD)")
def quick(
    strategy_name: str,
    data_file: str,
    cash: float,
    commission: float,
    start_date: Optional[str],
    end_date: Optional[str],
):
    """
    Quick backtest without config file.
    """
    strategy_class = _load_strategy_class(strategy_name)
    click.echo(f"\nRunning quick backtest: {strategy_class.__name__}")
    
    result = run_backtest(
        strategy_class=strategy_class,
        data_source=data_file,
        cash=cash,
        commission=commission,
        start_date=start_date,
        end_date=end_date,
    )
    click.echo(f"Final Value: ${result.final_value:,.2f} ({result.return_pct:+.2f}%)")


def _load_strategy_class(class_path: str):
    if "." not in class_path:
        try:
            from ai_trader.backtesting.strategies import classic
            if hasattr(classic, class_path): return getattr(classic, class_path)
        except (ImportError, AttributeError): pass
        try:
            from ai_trader.backtesting.strategies import portfolio
            if hasattr(portfolio, class_path): return getattr(portfolio, class_path)
        except (ImportError, AttributeError): pass
        raise ValueError(f"Strategy not found: {class_path}")

    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _get_strategies_from_module(module):
    import inspect
    from ai_trader.backtesting.strategies.base import BaseStrategy

    strategies = []
    for name, obj in inspect.getmembers(module):
        if (
            inspect.isclass(obj)
            and issubclass(obj, BaseStrategy)
            and obj is not BaseStrategy
            and not name.startswith("_")
        ):
            strategies.append((name, obj))
    return sorted(strategies, key=lambda x: x[0])


if __name__ == "__main__":
    cli()
