"""
Subnet Token Paper Trading Validator

WHAT I'M MEASURING:
    Paper trading performance on Bittensor subnet tokens.

MINER INTERFACE:
    Miners provide Docker containers with a long-running Actor class:
    
    class Actor:
        def __init__(self):
            '''
            Initialize your trading agent.
            
            Environment variables available:
                SUBTENSOR_NETWORK: "finney", "test", or "local"
                SUBTENSOR_ENDPOINT: WebSocket endpoint for chain queries
                CHUTES_API_KEY: For LLM inference via Chutes (optional)
            
            You have full access to query the chain:
                - Subnet prices, pool depths, volumes
                - Metagraph data
                - Historical blocks
                - Any on-chain data
            
            Initialize your models, state, and strategy here.
            This runs ONCE when your container starts.
            '''
            self.subtensor = bt.Subtensor(network=os.getenv("SUBTENSOR_NETWORK"))
            # Your initialization...
        
        async def step(self, block_info: dict) -> dict:
            '''
            Called EVERY BLOCK by the validator.
            
            Your container stays running - maintain state between calls.
            Pull whatever chain data you need. Make trading decisions.
            
            Args:
                block_info: {
                    "block": int,           # current block number
                    "timestamp": int,       # unix timestamp
                    "portfolio": {          # YOUR current paper portfolio
                        "tao": float,
                        "positions": {netuid: alpha_amount, ...}
                    },
                    "total_value_tao": float
                }
            
            Returns:
                {
                    "orders": [
                        {
                            "netuid": int,
                            "side": "buy" | "sell",
                            "amount": float  # TAO for buys, Alpha for sells
                        },
                        ...
                    ]
                }
            
            You can query the chain yourself for:
                - subtensor.get_subnet_prices()
                - subtensor.all_subnets()
                - subtensor.metagraph(netuid)
                - Any other chain data
            '''

CONTAINER LIFECYCLE:
    - Containers are loaded ONCE and kept running for extended periods
    - step() is called every block (~12 seconds)
    - Containers may be restarted on failure or version update
    - Design for persistence: save critical state, handle restarts gracefully

SCORING:
    - Sharpe ratio of paper portfolio returns
    - EMA smoothing for consistent performance
    - Higher risk-adjusted returns = higher weight

GROUND TRUTH:
    Real subnet token prices from chain (objective, external).
"""

import os
import sys
import time
import asyncio
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

import click
import bittensor as bt
from bittensor_wallet import Wallet

# Basilica/Affinetes for container execution
try:
    import affinetes as af_env
    BASILICA_AVAILABLE = True
except ImportError:
    BASILICA_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

HEARTBEAT_TIMEOUT = 600  # seconds before process restart
STARTING_TAO = 100.0  # initial paper portfolio value
MAX_ORDER_FRACTION = 0.25  # max fraction of portfolio per order
SLIPPAGE_FACTOR = 0.001  # 0.1% slippage per trade
RETURNS_WINDOW = 100  # number of returns for Sharpe calculation
EMA_ALPHA = 0.1  # EMA smoothing factor
CONTAINER_STEP_TIMEOUT = 10  # seconds for step() call
CONTAINER_INIT_TIMEOUT = 60  # seconds for container initialization
CONTAINER_RESTART_COOLDOWN = 300  # seconds before restarting failed container
MAX_CONSECUTIVE_FAILURES = 5  # failures before marking container dead

# Execution mode: "basilica" for production, "docker" for local dev
EXECUTION_MODE = os.getenv("EXECUTION_MODE", "basilica")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Portfolio:
    """Paper trading portfolio for a miner."""
    tao: float = STARTING_TAO
    positions: dict = field(default_factory=dict)  # netuid -> alpha amount
    value_history: list = field(default_factory=list)  # historical values in TAO
    last_update_block: int = 0
    
    def total_value(self, prices: dict[int, float]) -> float:
        """Calculate total portfolio value in TAO."""
        value = self.tao
        for netuid, alpha_amount in self.positions.items():
            if netuid in prices and prices[netuid] > 0:
                value += alpha_amount * prices[netuid]
        return value
    
    def to_dict(self, prices: dict[int, float]) -> dict:
        """Convert to dict for miner interface."""
        return {
            "tao": self.tao,
            "positions": dict(self.positions),
        }


@dataclass 
class MinerState:
    """Tracking state for a miner."""
    uid: int
    hotkey: str
    portfolio: Portfolio = field(default_factory=Portfolio)
    image_url: Optional[str] = None
    ema_score: float = 0.0
    consecutive_failures: int = 0
    last_orders: list = field(default_factory=list)
    
    # Container lifecycle
    container: Optional[object] = None  # af_env instance
    container_loaded_at: float = 0
    container_version: Optional[str] = None  # track image version
    last_failure_time: float = 0
    is_initializing: bool = False


# =============================================================================
# TRADING SIMULATION
# =============================================================================

def execute_orders(
    portfolio: Portfolio,
    orders: list[dict],
    prices: dict[int, float],
) -> list[dict]:
    """
    Execute paper trades with slippage simulation.
    
    Returns list of executed trades with details.
    """
    executed = []
    
    for order in orders:
        try:
            netuid = order.get("netuid")
            side = order.get("side", "").lower()
            amount = float(order.get("amount", 0))
            
            if netuid not in prices or prices[netuid] <= 0:
                continue
            if side not in ("buy", "sell"):
                continue
            if amount <= 0:
                continue
                
            price = prices[netuid]
            
            # Enforce position limits
            max_amount = portfolio.total_value(prices) * MAX_ORDER_FRACTION
            amount = min(amount, max_amount)
            
            if side == "buy":
                # Buying Alpha with TAO
                if amount > portfolio.tao:
                    amount = portfolio.tao
                if amount <= 0:
                    continue
                    
                # Apply slippage (price increases when buying)
                effective_price = price * (1 + SLIPPAGE_FACTOR)
                alpha_received = amount / effective_price
                
                portfolio.tao -= amount
                portfolio.positions[netuid] = portfolio.positions.get(netuid, 0) + alpha_received
                
                executed.append({
                    "netuid": netuid,
                    "side": "buy",
                    "tao_spent": amount,
                    "alpha_received": alpha_received,
                    "price": effective_price,
                })
                
            else:  # sell
                # Selling Alpha for TAO (amount is in Alpha)
                current_alpha = portfolio.positions.get(netuid, 0)
                alpha_to_sell = min(amount, current_alpha)
                if alpha_to_sell <= 0:
                    continue
                    
                # Apply slippage (price decreases when selling)
                effective_price = price * (1 - SLIPPAGE_FACTOR)
                tao_received = alpha_to_sell * effective_price
                
                portfolio.positions[netuid] = current_alpha - alpha_to_sell
                portfolio.tao += tao_received
                
                executed.append({
                    "netuid": netuid,
                    "side": "sell",
                    "alpha_sold": alpha_to_sell,
                    "tao_received": tao_received,
                    "price": effective_price,
                })
                
        except Exception as e:
            logger.debug(f"Order execution error: {e}")
            continue
    
    return executed


def calculate_sharpe(returns: list[float]) -> float:
    """Calculate Sharpe ratio from returns."""
    if len(returns) < 2:
        return 0.0
    
    import statistics
    mean_return = statistics.mean(returns)
    std_return = statistics.stdev(returns)
    
    if std_return == 0:
        return mean_return * 10 if mean_return > 0 else 0.0
    
    return mean_return / std_return


def calculate_returns(value_history: list[float]) -> list[float]:
    """Calculate percentage returns from value history."""
    if len(value_history) < 2:
        return []
    
    returns = []
    for i in range(1, len(value_history)):
        if value_history[i-1] > 0:
            ret = (value_history[i] - value_history[i-1]) / value_history[i-1]
            returns.append(ret)
    
    return returns


# =============================================================================
# CONTAINER LIFECYCLE MANAGEMENT
# =============================================================================

async def load_container(
    miner: MinerState,
    network: str,
    subtensor_endpoint: str,
    mode: str = "basilica",
) -> bool:
    """
    Load a miner's container and keep it running.
    
    Returns True if successfully loaded.
    """
    if not miner.image_url:
        return False
    
    if not BASILICA_AVAILABLE:
        logger.warning("Affinetes not available")
        return False
    
    # Check if already loaded with same version
    if miner.container and miner.container_version == miner.image_url:
        return True
    
    # Cleanup old container if exists
    if miner.container:
        try:
            await miner.container.cleanup()
        except Exception:
            pass
        miner.container = None
    
    miner.is_initializing = True
    
    try:
        logger.info(f"Loading container for miner {miner.uid}: {miner.image_url}")
        
        env = af_env.load_env(
            mode=mode,
            image=miner.image_url,
            cpu_limit="2000m",
            mem_limit="4Gi",
            env_vars={
                "SUBTENSOR_NETWORK": network,
                "SUBTENSOR_ENDPOINT": subtensor_endpoint,
                "CHUTES_API_KEY": os.getenv("CHUTES_API_KEY", ""),
            },
            # Keep container alive
            keep_alive=True,
        )
        
        # Wait for initialization (Actor.__init__ runs)
        await asyncio.wait_for(
            env.wait_ready(),
            timeout=CONTAINER_INIT_TIMEOUT
        )
        
        miner.container = env
        miner.container_loaded_at = time.time()
        miner.container_version = miner.image_url
        miner.consecutive_failures = 0
        
        logger.info(f"Container loaded for miner {miner.uid}")
        return True
        
    except asyncio.TimeoutError:
        logger.warning(f"Miner {miner.uid} container init timeout")
        miner.last_failure_time = time.time()
        return False
    except Exception as e:
        logger.warning(f"Miner {miner.uid} container load error: {e}")
        miner.last_failure_time = time.time()
        return False
    finally:
        miner.is_initializing = False


async def execute_step(
    miner: MinerState,
    block_info: dict,
    prices: dict[int, float],
) -> Optional[dict]:
    """
    Execute one step on a miner's running container.
    
    Returns orders dict or None on failure.
    """
    if not miner.container:
        return None
    
    try:
        # Build block info with portfolio state
        step_input = {
            "block": block_info["block"],
            "timestamp": block_info["timestamp"],
            "portfolio": miner.portfolio.to_dict(prices),
            "total_value_tao": miner.portfolio.total_value(prices),
        }
        
        result = await asyncio.wait_for(
            miner.container.step(block_info=step_input),
            timeout=CONTAINER_STEP_TIMEOUT
        )
        
        miner.consecutive_failures = 0
        return result
        
    except asyncio.TimeoutError:
        logger.debug(f"Miner {miner.uid} step timeout")
        miner.consecutive_failures += 1
        return None
    except Exception as e:
        logger.debug(f"Miner {miner.uid} step error: {e}")
        miner.consecutive_failures += 1
        
        # Mark container as dead if too many failures
        if miner.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            logger.warning(f"Miner {miner.uid} container marked dead after {miner.consecutive_failures} failures")
            try:
                await miner.container.cleanup()
            except Exception:
                pass
            miner.container = None
            miner.last_failure_time = time.time()
        
        return None


async def cleanup_container(miner: MinerState):
    """Cleanup a miner's container."""
    if miner.container:
        try:
            await miner.container.cleanup()
        except Exception as e:
            logger.debug(f"Container cleanup error for miner {miner.uid}: {e}")
        finally:
            miner.container = None


# =============================================================================
# PRICE FETCHING
# =============================================================================

def get_subnet_prices(subtensor: bt.Subtensor) -> dict[int, float]:
    """Fetch current prices for all subnets."""
    prices = {}
    try:
        all_prices = subtensor.get_subnet_prices()
        if all_prices:
            for netuid, price in all_prices.items():
                if price and float(price) > 0:
                    prices[int(netuid)] = float(price)
    except Exception as e:
        logger.error(f"Error fetching subnet prices: {e}")
    
    return prices


# =============================================================================
# SCORING AND WEIGHTS
# =============================================================================

def calculate_weights(miners: dict[int, MinerState]) -> tuple[list[int], list[float]]:
    """
    Calculate normalized weights from miner scores.
    
    Uses softmax normalization for smooth weight distribution.
    """
    if not miners:
        return [], []
    
    uids = []
    scores = []
    
    for uid, miner in miners.items():
        uids.append(uid)
        scores.append(max(miner.ema_score, 0))
    
    # Normalize
    total = sum(scores)
    if total == 0:
        weights = [1.0 / len(uids)] * len(uids)
    else:
        weights = [s / total for s in scores]
    
    return uids, weights


def update_miner_scores(miners: dict[int, MinerState], prices: dict[int, float]):
    """Update EMA scores based on current Sharpe ratios."""
    for uid, miner in miners.items():
        # Record current portfolio value
        current_value = miner.portfolio.total_value(prices)
        miner.portfolio.value_history.append(current_value)
        
        # Keep only recent history
        if len(miner.portfolio.value_history) > RETURNS_WINDOW + 1:
            miner.portfolio.value_history = miner.portfolio.value_history[-RETURNS_WINDOW - 1:]
        
        # Calculate Sharpe ratio
        returns = calculate_returns(miner.portfolio.value_history)
        sharpe = calculate_sharpe(returns)
        
        # Normalize Sharpe to [0, 1] range
        normalized_score = max(0, min(1, (sharpe + 2) / 4))
        
        # EMA update
        miner.ema_score = EMA_ALPHA * normalized_score + (1 - EMA_ALPHA) * miner.ema_score


# =============================================================================
# MINER DISCOVERY
# =============================================================================

def get_miner_image_urls(
    subtensor: bt.Subtensor,
    netuid: int,
) -> dict[int, str]:
    """Read miner Docker image URLs from chain commitments."""
    image_urls = {}
    
    try:
        commitments = subtensor.get_all_commitments(netuid)
        if commitments:
            for uid, commitment in commitments.items():
                if commitment and isinstance(commitment, str):
                    # Expect format: docker.io/user/image:tag
                    if "/" in commitment:
                        image_urls[int(uid)] = commitment
    except Exception as e:
        logger.debug(f"Error reading commitments: {e}")
    
    return image_urls


# =============================================================================
# HEARTBEAT MONITOR
# =============================================================================

def heartbeat_monitor(last_heartbeat: list, stop_event: threading.Event):
    """Monitor for validator liveness, restart if stuck."""
    while not stop_event.is_set():
        time.sleep(5)
        if time.time() - last_heartbeat[0] > HEARTBEAT_TIMEOUT:
            logger.error("No heartbeat detected. Restarting process.")
            logging.shutdown()
            os.execv(sys.executable, [sys.executable] + sys.argv)


# =============================================================================
# MAIN VALIDATOR
# =============================================================================

@click.command()
@click.option(
    "--network",
    default=lambda: os.getenv("NETWORK", "finney"),
    help="Network to connect to (finney, test, local)",
)
@click.option(
    "--netuid",
    type=int,
    default=lambda: int(os.getenv("NETUID", "1")),
    help="Subnet netuid",
)
@click.option(
    "--coldkey",
    default=lambda: os.getenv("WALLET_NAME", "default"),
    help="Wallet name",
)
@click.option(
    "--hotkey",
    default=lambda: os.getenv("HOTKEY_NAME", "default"),
    help="Hotkey name",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default=lambda: os.getenv("LOG_LEVEL", "INFO"),
    help="Logging level",
)
@click.option(
    "--execution-mode",
    type=click.Choice(["basilica", "docker"], case_sensitive=False),
    default=lambda: os.getenv("EXECUTION_MODE", "basilica"),
    help="Container execution mode",
)
@click.option(
    "--subtensor-endpoint",
    default=lambda: os.getenv("SUBTENSOR_ENDPOINT", ""),
    help="WebSocket endpoint for miner chain access",
)
def main(
    network: str,
    netuid: int,
    coldkey: str,
    hotkey: str,
    log_level: str,
    execution_mode: str,
    subtensor_endpoint: str,
):
    """
    Run the Paper Trading subnet validator.
    
    Miners submit Docker containers with long-running trading agents.
    Containers are loaded once and step() is called every block.
    Validator maintains paper portfolios and scores based on returns.
    """
    logging.getLogger().setLevel(getattr(logging, log_level.upper()))
    logger.info(f"Starting Paper Trading validator on network={network}, netuid={netuid}")
    logger.info(f"Execution mode: {execution_mode}")
    
    # Heartbeat setup
    last_heartbeat = [time.time()]
    stop_event = threading.Event()
    heartbeat_thread = threading.Thread(
        target=heartbeat_monitor,
        args=(last_heartbeat, stop_event),
        daemon=True
    )
    heartbeat_thread.start()
    
    # For passing to containers
    if not subtensor_endpoint:
        if network == "finney":
            subtensor_endpoint = "wss://entrypoint-finney.opentensor.ai:443"
        elif network == "test":
            subtensor_endpoint = "wss://test.finney.opentensor.ai:443"
        else:
            subtensor_endpoint = "ws://127.0.0.1:9944"
    
    async def run_validator():
        # Initialize wallet, subtensor, metagraph
        wallet = Wallet(name=coldkey, hotkey=hotkey)
        subtensor = bt.Subtensor(network=network)
        metagraph = bt.Metagraph(netuid=netuid, network=network)
        
        # Sync metagraph
        metagraph.sync(subtensor=subtensor)
        logger.info(f"Metagraph synced: {metagraph.n} neurons at block {metagraph.block}")
        
        # Verify registration
        my_hotkey = wallet.hotkey.ss58_address
        if my_hotkey not in metagraph.hotkeys:
            logger.error(f"Hotkey {my_hotkey} not registered on netuid {netuid}")
            stop_event.set()
            return
        my_uid = metagraph.hotkeys.index(my_hotkey)
        logger.info(f"Validator UID: {my_uid}")
        
        # Get subnet hyperparameters
        hyperparams = subtensor.get_subnet_hyperparameters(netuid)
        tempo = hyperparams.tempo
        logger.info(f"Subnet tempo: {tempo} blocks")
        
        # Initialize miner states
        miners: dict[int, MinerState] = {}
        for uid in range(metagraph.n):
            if uid == my_uid:
                continue
            miners[uid] = MinerState(
                uid=uid,
                hotkey=metagraph.hotkeys[uid],
            )
        
        last_weight_block = 0
        last_block = 0
        
        try:
            # Main validator loop
            while not stop_event.is_set():
                try:
                    loop_start = time.time()
                    
                    # Sync metagraph
                    metagraph.sync(subtensor=subtensor, lite=True)
                    current_block = subtensor.get_current_block()
                    
                    # Update heartbeat
                    last_heartbeat[0] = time.time()
                    
                    # Skip if same block
                    if current_block == last_block:
                        await asyncio.sleep(1)
                        continue
                    last_block = current_block
                    
                    # Update miner list (handle new registrations)
                    for uid in range(metagraph.n):
                        if uid == my_uid:
                            continue
                        if uid not in miners:
                            miners[uid] = MinerState(
                                uid=uid,
                                hotkey=metagraph.hotkeys[uid],
                            )
                    
                    # Fetch miner container URLs from commitments
                    image_urls = get_miner_image_urls(subtensor, netuid)
                    for uid, url in image_urls.items():
                        if uid in miners:
                            old_url = miners[uid].image_url
                            miners[uid].image_url = url
                            # Detect version change
                            if old_url and old_url != url:
                                logger.info(f"Miner {uid} updated image: {url}")
                    
                    # Get current subnet prices
                    prices = get_subnet_prices(subtensor)
                    
                    if not prices:
                        logger.warning("No subnet prices available")
                        await asyncio.sleep(12)
                        continue
                    
                    logger.debug(f"Block {current_block}: {len(prices)} subnets with prices")
                    
                    # Load containers for miners that need them
                    load_tasks = []
                    for uid, miner in miners.items():
                        if not miner.image_url:
                            continue
                        
                        # Skip if initializing
                        if miner.is_initializing:
                            continue
                        
                        # Check if container needs loading
                        needs_load = (
                            miner.container is None or
                            miner.container_version != miner.image_url
                        )
                        
                        # Respect cooldown after failures
                        if needs_load and miner.last_failure_time > 0:
                            if time.time() - miner.last_failure_time < CONTAINER_RESTART_COOLDOWN:
                                continue
                        
                        if needs_load:
                            load_tasks.append(
                                load_container(
                                    miner,
                                    network,
                                    subtensor_endpoint,
                                    execution_mode,
                                )
                            )
                    
                    if load_tasks:
                        logger.info(f"Loading {len(load_tasks)} containers")
                        await asyncio.gather(*load_tasks, return_exceptions=True)
                    
                    # Execute step on all running containers
                    block_info = {
                        "block": current_block,
                        "timestamp": int(time.time()),
                    }
                    
                    active_miners = [m for m in miners.values() if m.container is not None]
                    
                    if active_miners:
                        logger.info(f"Block {current_block}: Stepping {len(active_miners)} containers")
                        
                        # Execute steps concurrently
                        step_tasks = [
                            execute_step(miner, block_info, prices)
                            for miner in active_miners
                        ]
                        results = await asyncio.gather(*step_tasks, return_exceptions=True)
                        
                        # Process results
                        for miner, result in zip(active_miners, results):
                            if isinstance(result, Exception):
                                logger.debug(f"Miner {miner.uid} step exception: {result}")
                                continue
                            
                            if result is None:
                                continue
                            
                            orders = result.get("orders", [])
                            if orders:
                                executed = execute_orders(
                                    miner.portfolio,
                                    orders,
                                    prices,
                                )
                                miner.last_orders = executed
                                miner.portfolio.last_update_block = current_block
                                
                                if executed:
                                    logger.debug(
                                        f"Miner {miner.uid}: {len(executed)} trades, "
                                        f"value: {miner.portfolio.total_value(prices):.2f} TAO"
                                    )
                    
                    # Update scores for all miners
                    update_miner_scores(miners, prices)
                    
                    # Set weights once per tempo
                    blocks_since_last = current_block - last_weight_block
                    if blocks_since_last >= tempo:
                        logger.info(f"Block {current_block}: Setting weights")
                        
                        uids, weights = calculate_weights(miners)
                        
                        if uids:
                            # Log top performers
                            sorted_miners = sorted(
                                miners.items(),
                                key=lambda x: x[1].ema_score,
                                reverse=True
                            )[:5]
                            logger.info("Top 5 miners:")
                            for uid, miner in sorted_miners:
                                value = miner.portfolio.total_value(prices)
                                container_status = "running" if miner.container else "stopped"
                                logger.info(
                                    f"  UID {uid}: score={miner.ema_score:.4f}, "
                                    f"value={value:.2f} TAO, container={container_status}"
                                )
                            
                            success = subtensor.set_weights(
                                wallet=wallet,
                                netuid=netuid,
                                uids=uids,
                                weights=weights,
                                wait_for_inclusion=True,
                                wait_for_finalization=False,
                            )
                            
                            if success:
                                logger.info(f"Set weights for {len(uids)} miners")
                                last_weight_block = current_block
                            else:
                                logger.warning("Failed to set weights")
                    
                    # Sleep until next block
                    elapsed = time.time() - loop_start
                    await asyncio.sleep(max(0, 12 - elapsed))
                    
                except KeyboardInterrupt:
                    logger.info("Validator stopped by user")
                    break
                except Exception as e:
                    logger.error(f"Error in validator loop: {e}", exc_info=True)
                    await asyncio.sleep(12)
                    
        finally:
            # Cleanup all containers
            logger.info("Cleaning up containers...")
            cleanup_tasks = [cleanup_container(m) for m in miners.values()]
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
    
    try:
        asyncio.run(run_validator())
    finally:
        stop_event.set()
        heartbeat_thread.join(timeout=2)


if __name__ == "__main__":
    main()
