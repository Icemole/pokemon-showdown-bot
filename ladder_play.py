import asyncio
import argparse
from tensorflow import keras
from poke_env.player_configuration import PlayerConfiguration
from poke_env.server_configuration import ShowdownServerConfiguration
from src.players import SelfPlayRLPlayer
import logging

logging.basicConfig(filename='ladder_play.log', encoding='utf-8', level=logging.DEBUG)


async def main():
    parser = argparse.ArgumentParser(description='Launch script to use a poke-env powered bot in a public ladder')
    parser.add_argument('--model', help='Path to model')
    parser.add_argument('--format',  help='Battle format')
    parser.add_argument('--battles', default=1, help='Number of battles to perform in the ladder')
    args = parser.parse_args()

    # Load trained model
    model = keras.models.load_model(args.model)

    # Instantiate rl trained player
    rl_player = SelfPlayRLPlayer(model=model, player_configuration=PlayerConfiguration("pleaseJorgea10", "arf2021"),
        server_configuration=ShowdownServerConfiguration, battle_format=args.format, start_timer_on_battle_start=True)

    # Playing 5 games on the ladder
    await rl_player.ladder(args.battles)

    # Print the rating of the player and its opponent after each battle
    for battle in rl_player.battles.values():
        logging.info(battle.rating, battle.opponent_rating)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())