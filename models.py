import torch
import torch.nn as nn
import sys


class SimpleModel(nn.Module):
    def __init__(self, n_actions):
        """
        Our embeddings have shape (1, 10), which affects our hidden layer
        dimension and output dimension
        Flattening resolves potential issues that would arise otherwise
        """
        super(SimpleModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(10, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, 64),
            nn.ELU(inplace=True),
            nn.Linear(64, n_actions),
        )

    def forward(self, state):
        return self.model(state)


class PokemonModel(nn.Module):
    def __init__(self, embedding_dim, max_values, pokemon_others_size):
        super(PokemonModel, self).__init__()
        # Define Embedding Layers
        # Input: (batch_size, 1)
        # Output: (batch_size, 1, embedding_dim)
        self.species_embedding = nn.Embedding(
            num_embeddings=max_values["species"],
            embedding_dim=embedding_dim,
        )
        self.item_embedding = nn.Embedding(
            num_embeddings=max_values["items"],
            embedding_dim=embedding_dim,
        )
        self.ability_embedding = nn.Embedding(
            num_embeddings=max_values["abilities"],
            embedding_dim=embedding_dim,
        )
        self.move_embedding = nn.Embedding(
            num_embeddings=max_values["moves"],
            embedding_dim=embedding_dim,
        )
        # Final embedding dim =
        # embedding_dim * num_things_to_get_embeddings_for
        # This number is = 5
        # 1 species, 1 item, 1 ability, 1 average ability, 1 average move
        in_features = (embedding_dim * 5) + pokemon_others_size
        self.model = nn.Linear(in_features=in_features, out_features=embedding_dim)

    def forward(self, state):
        # First dimension is batch size
        # First 10 vectors = things we need embeddings for
        # Rest is concatenated before passing through the dense layer
        pokemon_state = state[:, :10].int()
        pokemon_others_state = state[:, 10:]
        # Define Inputs
        pokemon_species, pokemon_item, pokemon_ability, pokemon_possible_ability1, pokemon_possible_ability2, pokemon_possible_ability3, pokemon_move1, pokemon_move2, pokemon_move3, pokemon_move4 = torch.chunk(pokemon_state, chunks=10, dim=-1)

        # Get Embeddings: (batch_size, embedding_dim)
        species_embedded = self.species_embedding(pokemon_species).squeeze(1)
        item_embedded = self.item_embedding(pokemon_item).squeeze(1)
        ability_embedded = self.ability_embedding(pokemon_ability).squeeze(1)

        # Average of Possible Abilities: (batch_size, embedding_dim)
        possible_ability1_embedded = self.ability_embedding(pokemon_possible_ability1)
        possible_ability2_embedded = self.ability_embedding(pokemon_possible_ability2)
        possible_ability3_embedded = self.ability_embedding(pokemon_possible_ability3)
        average_abilities_embedded = torch.stack(
            [
                possible_ability1_embedded,
                possible_ability2_embedded,
                possible_ability3_embedded,
            ]
        ).mean(dim=0).squeeze(1)

        # Average Embedding of 4 Moves: (batch_size, embedding_dim)
        move1_embedded = self.move_embedding(pokemon_move1)
        move2_embedded = self.move_embedding(pokemon_move2)
        move3_embedded = self.move_embedding(pokemon_move3)
        move4_embedded = self.move_embedding(pokemon_move4)
        average_moves_embedded = torch.stack(
            [
                move1_embedded,
                move2_embedded,
                move3_embedded,
                move4_embedded,
            ]
        ).mean(dim=0).squeeze(1)

        # Concatenate all Vectors: (batch_size, embedding_dim * 5 + other_dim)
        pokemon_embedded = torch.cat(
            [
                species_embedded,
                item_embedded,
                ability_embedded,
                average_abilities_embedded,
                average_moves_embedded,
                pokemon_others_state,
            ], dim=1
        )
        # Pass through model: (batch_size, embedding_dim)
        return self.model(pokemon_embedded).relu()


class TeamModel(nn.Module):
    def __init__(self, embedding_dim, pokemon_embedding_dim, team_size):
        super(TeamModel, self).__init__()
        # In the paper they use a MaxPool before passing the vector to
        # the layer but there's no information on the specifics of it.
        # So I'm just doing a concat before passing it to the linear layer.
        self.pokemon_team_embedding = nn.Linear(
            in_features=6 * pokemon_embedding_dim, out_features=embedding_dim
        )

        # Input features =
        # dim(pokemon_team_embedded) +
        # dim(active_pokemon_embedded) +
        # dim(team_conditions)
        in_features = embedding_dim + pokemon_embedding_dim + team_size
        self.model = nn.Linear(in_features=in_features, out_features=embedding_dim)

    def forward(self, pokemon, team_conditions):
        # pokemon is a list of 6 tensors
        # [t1, t2, t3, t4, t5, t6]
        # Where each tensor is (batch_size, pokemon_embedding_dim)
        active_pokemon_embedded = pokemon[0]
        pokemon = torch.cat([*pokemon], dim=-1)

        pokemon_team_embedded = self.pokemon_team_embedding(pokemon).relu()
        team_embedded = torch.cat(
            [pokemon_team_embedded, active_pokemon_embedded, team_conditions],
            dim=-1
        )
        return self.model(team_embedded).relu()


class BattleModel(nn.Module):
    def __init__(
        self,
        n_actions,
        pokemon_embedding_dim,
        team_embedding_dim,
        max_values_dict,
        state_length_dict
    ):
        """
        State: 
        [*player_pokemon1, *player_pokemon2, *player_pokemon3, 
        *player_pokemon4, *player_pokemon5, *player_pokemon6, player_state, 
        *opponent_pokemon1, *opponent_pokemon2, *opponent_pokemon3, 
        *opponent_pokemon4, *opponent_pokemon5, *opponent_pokemon6, 
        opponent_state, battle_state, action_mask]

        Where, each pokemon consists of 11 vectors:
        [pokemon_species, pokemon_item, pokemon_ability, 
        pokemon_possible_ability1, pokemon_possible_ability2, 
        pokemon_possible_ability3, pokemon_move1, pokemon_move2, 
        pokemon_move3, pokemon_move4, pokemon_others_state]

        In total: 
        12 Pokemon * 11 Vectors + 2 Team Vectors + 1 Team Vector + 1 Mask
        = State of size 136
        """
        super(BattleModel, self).__init__()
        self.pokemon_per_team = 6
        self.pokemon_state_length = state_length_dict["pokemon_state"] + state_length_dict["pokemon_others_state"]
        self.all_pokemon_state_length = self.pokemon_state_length * self.pokemon_per_team
        self.team_state_length = state_length_dict["team_state"]
        self.battle_state_length = state_length_dict["battle_state"]

        in_features = team_embedding_dim + team_embedding_dim + self.battle_state_length

        self.pokemon_model = PokemonModel(
            pokemon_embedding_dim, max_values_dict, state_length_dict["pokemon_others_state"]
        )
        self.team_model = TeamModel(
            team_embedding_dim, pokemon_embedding_dim, state_length_dict["team_state"]
        )
        self.model = nn.Linear(in_features=in_features, out_features=n_actions)

    def process(self, batch):
        """
        Since we have a complex input torch can't batch it up properly.
        So we handle this ourselves.
        """
        if batch.ndim == 1:
            return batch.unsqueeze(0)
        else:
            return batch

    def forward(self, state):
        # Convert state to (batch_size, state) if not already in it
        state = self.process(state)
        # Segment out the different parts of the state
        # Applying this only on the 1st dimension (IE not the batch dim)
        player_pokemon_state = state[:, 0:self.all_pokemon_state_length]
        x = self.all_pokemon_state_length
        player_state = state[:, x:x + self.team_state_length]
        x = x + self.team_state_length
        opponent_pokemon_state = state[:, x: x + self.all_pokemon_state_length]
        x = x + self.all_pokemon_state_length
        opponent_state = state[:, x:x + self.team_state_length]
        x = x + self.team_state_length
        battle_state = state[:, x: x + self.battle_state_length]
        x = x + self.battle_state_length
        action_mask = state[:, x:]

        # Get embeddings for each individual pokemon
        player_pokemon = []
        opponent_pokemon = []
        start = 0
        end = start + self.pokemon_state_length
        for i in range(6):
            if i == 0:
                # TODO: Active pokemon stuff
                pass
            pokemon = self.pokemon_model(player_pokemon_state[:, start:end])
            player_pokemon.append(pokemon)

            pokemon = self.pokemon_model(opponent_pokemon_state[:, start:end])
            opponent_pokemon.append(pokemon)

            start = end
            end = start + self.pokemon_state_length

        player_team = self.team_model(player_pokemon, player_state)
        opponent_team = self.team_model(opponent_pokemon, opponent_state)

        # TODO: Concat battle embedding with vectors for each action.
        # TODO: Concat each action-specific vector so we have (22, len)
        # TODO: At present we just predict all actions from a single state. We should be doing the action concatenation thing from the paper.
        battle_state = torch.cat([player_team, opponent_team, battle_state], dim=-1)
        actions = self.model(battle_state)
        actions = actions + action_mask
        return nn.functional.softmax(actions, dim=-1)