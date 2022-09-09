"""
Collection of utility functions used in both RL Agent Training & Meta Discovery.
"""
from collections import namedtuple

def generate_server_configuration(port):
    ServerConfiguration = namedtuple(
        "ServerConfiguration", ["server_url", "authentication_url"]
    )
    LocalhostServerConfiguration = ServerConfiguration(
        f"localhost:{port}", "https://play.pokemonshowdown.com/action.php?"
    )
    return LocalhostServerConfiguration