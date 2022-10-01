import torch
import torch.nn as nn
import sys


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

    def forward(self, state, return_moveset=False):
        # First dimension is batch size
        # First 11 vectors = things we need embeddings for
        # Rest is concatenated before passing through the dense layer
        pokemon_state = state[:, :11].int()
        pokemon_others_state = state[:, 11:]
        # Define Inputs
        (
            pokemon_species,
            pokemon_item,
            pokemon_ability,
            pokemon_possible_ability1,
            pokemon_possible_ability2,
            pokemon_possible_ability3,
            pokemon_possible_ability4,
            pokemon_move1,
            pokemon_move2,
            pokemon_move3,
            pokemon_move4,
        ) = torch.chunk(pokemon_state, chunks=11, dim=-1)

        # Get Embeddings: (batch_size, embedding_dim)
        species_embedded = self.species_embedding(pokemon_species).squeeze(1)
        item_embedded = self.item_embedding(pokemon_item).squeeze(1)
        ability_embedded = self.ability_embedding(pokemon_ability).squeeze(1)

        # Average of Possible Abilities: (batch_size, embedding_dim)
        possible_ability1_embedded = self.ability_embedding(pokemon_possible_ability1)
        possible_ability2_embedded = self.ability_embedding(pokemon_possible_ability2)
        possible_ability3_embedded = self.ability_embedding(pokemon_possible_ability3)
        possible_ability4_embedded = self.ability_embedding(pokemon_possible_ability4)
        average_abilities_embedded = (
            torch.stack(
                [
                    possible_ability1_embedded,
                    possible_ability2_embedded,
                    possible_ability3_embedded,
                    possible_ability4_embedded,
                ]
            )
            .mean(dim=0)
            .squeeze(1)
        )

        # Average Embedding of 4 Moves: (batch_size, embedding_dim)
        move1_embedded = self.move_embedding(pokemon_move1).squeeze(1)
        move2_embedded = self.move_embedding(pokemon_move2).squeeze(1)
        move3_embedded = self.move_embedding(pokemon_move3).squeeze(1)
        move4_embedded = self.move_embedding(pokemon_move4).squeeze(1)
        average_moves_embedded = torch.stack(
            [
                move1_embedded,
                move2_embedded,
                move3_embedded,
                move4_embedded,
            ]
        ).mean(dim=0)

        # Concatenate all Vectors: (batch_size, embedding_dim * 5 + other_dim)
        pokemon_embedded = torch.cat(
            [
                species_embedded,
                item_embedded,
                ability_embedded,
                average_abilities_embedded,
                average_moves_embedded,
                pokemon_others_state,
            ],
            dim=1,
        )
        # Pass through model: (batch_size, embedding_dim)
        out = self.model(pokemon_embedded).relu()
        if return_moveset:
            return out, move1_embedded, move2_embedded, move3_embedded, move4_embedded
        else:
            return out


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

    def forward(self, pokemon, team_conditions, active_pokemon_index):
        # pokemon is a list of 6 tensors
        # [t1, t2, t3, t4, t5, t6]
        # Where each tensor is (batch_size, pokemon_embedding_dim)
        batch_index = torch.arange(0, active_pokemon_index.shape[0])
        active_pokemon_embedded = torch.stack(pokemon)[
            active_pokemon_index, batch_index
        ]
        pokemon = torch.cat([*pokemon], dim=-1)

        pokemon_team_embedded = self.pokemon_team_embedding(pokemon).relu()
        team_embedded = torch.cat(
            [pokemon_team_embedded, active_pokemon_embedded, team_conditions], dim=-1
        )
        return self.model(team_embedded).relu()


class BattleModel(nn.Module):
    def __init__(
        self,
        n_actions,
        pokemon_embedding_dim,
        team_embedding_dim,
        max_values_dict,
        state_length_dict,
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
        self.pokemon_state_length = (
            state_length_dict["pokemon_state"]
            + state_length_dict["pokemon_others_state"]
        )
        self.all_pokemon_state_length = (
            self.pokemon_state_length * self.pokemon_per_team
        )
        self.team_state_length = state_length_dict["team_state"]
        self.battle_state_length = state_length_dict["battle_state"]

        in_features = team_embedding_dim + team_embedding_dim + self.battle_state_length

        self.pokemon_model = PokemonModel(
            pokemon_embedding_dim,
            max_values_dict,
            state_length_dict["pokemon_others_state"],
        )
        self.team_model = TeamModel(
            team_embedding_dim, pokemon_embedding_dim, state_length_dict["team_state"]
        )
        # (batch_size, in_features) -> (batch_size, n_actions)
        self.battle_state_model = nn.Linear(
            in_features=in_features, out_features=n_actions
        )
        # (batch_size, n_actions, pokemon_embedding_dim + 1) ->
        # (batch_size, n_actions, 1)
        self.model = nn.Linear(in_features=1 + pokemon_embedding_dim, out_features=1)
        # Save these variables for (potential) future use
        self.pokemon_embedding_dim = pokemon_embedding_dim
        self.team_embedding_dim = team_embedding_dim

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
        player_pokemon_state = state[:, 0 : self.all_pokemon_state_length]
        x = self.all_pokemon_state_length
        player_state = state[:, x : x + self.team_state_length]
        x = x + self.team_state_length
        opponent_pokemon_state = state[:, x : x + self.all_pokemon_state_length]
        x = x + self.all_pokemon_state_length
        opponent_state = state[:, x : x + self.team_state_length]
        x = x + self.team_state_length
        battle_state = state[:, x : x + self.battle_state_length]
        x = x + self.battle_state_length
        player_active_pokemon_index = state[:, x].long()
        x = x + 1
        opponent_active_pokemon_index = state[:, x].long()

        # Get embeddings for each individual pokemon
        player_pokemon = []
        opponent_pokemon = []
        active_move1_batch = torch.zeros(
            state.shape[0], self.pokemon_embedding_dim, device=state.device
        )
        active_move2_batch = torch.zeros(
            state.shape[0], self.pokemon_embedding_dim, device=state.device
        )
        active_move3_batch = torch.zeros(
            state.shape[0], self.pokemon_embedding_dim, device=state.device
        )
        active_move4_batch = torch.zeros(
            state.shape[0], self.pokemon_embedding_dim, device=state.device
        )
        start = 0
        end = start + self.pokemon_state_length
        for i in range(6):
            (
                pokemon,
                active_move1,
                active_move2,
                active_move3,
                active_move4,
            ) = self.pokemon_model(
                player_pokemon_state[:, start:end], return_moveset=True
            )
            player_pokemon.append(pokemon)

            # Keep track of the active moves
            active_move1_batch[player_active_pokemon_index == i] = active_move1[
                player_active_pokemon_index == i
            ]
            active_move2_batch[player_active_pokemon_index == i] = active_move2[
                player_active_pokemon_index == i
            ]
            active_move3_batch[player_active_pokemon_index == i] = active_move3[
                player_active_pokemon_index == i
            ]
            active_move4_batch[player_active_pokemon_index == i] = active_move4[
                player_active_pokemon_index == i
            ]

            pokemon = self.pokemon_model(opponent_pokemon_state[:, start:end])
            opponent_pokemon.append(pokemon)

            start = end
            end = start + self.pokemon_state_length

        player_team = self.team_model(
            player_pokemon, player_state, player_active_pokemon_index
        )
        opponent_team = self.team_model(
            opponent_pokemon, opponent_state, opponent_active_pokemon_index
        )

        battle_state = torch.cat([player_team, opponent_team, battle_state], dim=-1)
        battle_state = self.battle_state_model(battle_state)
        # 22 Actions:
        # 0-4, 4-8, 8-12, 12-16 are for moves.
        # Moves, Z-Moves, Mega Evolved Moves, Dynamaxed Moves
        # 16-22 = 6 Switches
        # TODO: Multiply/Add move vector by the relevant bonus (z/mega/dyna)
        emphasis_vector = torch.stack(
            [
                active_move1,
                active_move2,
                active_move3,
                active_move4,
                active_move1,
                active_move2,
                active_move3,
                active_move4,
                active_move1,
                active_move2,
                active_move3,
                active_move4,
                active_move1,
                active_move2,
                active_move3,
                active_move4,
                player_pokemon[0],
                player_pokemon[1],
                player_pokemon[2],
                player_pokemon[3],
                player_pokemon[4],
                player_pokemon[5],
            ],
            dim=1,
        )
        battle_state = torch.cat([battle_state.unsqueeze(-1), emphasis_vector], dim=-1)
        actions = self.model(battle_state).squeeze(-1)
        return actions


class ActorCriticBattleModel(nn.Module):
    def __init__(
        self,
        n_actions,
        pokemon_embedding_dim,
        team_embedding_dim,
        max_values_dict,
        state_length_dict,
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
        super(ActorCriticBattleModel, self).__init__()
        self.pokemon_per_team = 6
        self.pokemon_state_length = (
            state_length_dict["pokemon_state"]
            + state_length_dict["pokemon_others_state"]
        )
        self.all_pokemon_state_length = (
            self.pokemon_state_length * self.pokemon_per_team
        )
        self.team_state_length = state_length_dict["team_state"]
        self.battle_state_length = state_length_dict["battle_state"]

        in_features = team_embedding_dim + team_embedding_dim + self.battle_state_length

        self.pokemon_model = PokemonModel(
            pokemon_embedding_dim,
            max_values_dict,
            state_length_dict["pokemon_others_state"],
        )
        self.team_model = TeamModel(
            team_embedding_dim, pokemon_embedding_dim, state_length_dict["team_state"]
        )
        # (batch_size, in_features) -> (batch_size, n_actions)
        self.policy_head = nn.Linear(in_features=in_features, out_features=n_actions)
        self.value_head = nn.Linear(in_features=in_features, out_features=1)
        # (batch_size, n_actions, pokemon_embedding_dim + 1) ->
        # (batch_size, n_actions, 1)
        self.policy_emphasis_head = nn.Linear(
            in_features=1 + pokemon_embedding_dim, out_features=1
        )
        # Save these variables for (potential) future use
        self.pokemon_embedding_dim = pokemon_embedding_dim
        self.team_embedding_dim = team_embedding_dim

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
        player_pokemon_state = state[:, 0 : self.all_pokemon_state_length]
        x = self.all_pokemon_state_length
        player_state = state[:, x : x + self.team_state_length]
        x = x + self.team_state_length
        opponent_pokemon_state = state[:, x : x + self.all_pokemon_state_length]
        x = x + self.all_pokemon_state_length
        opponent_state = state[:, x : x + self.team_state_length]
        x = x + self.team_state_length
        battle_state = state[:, x : x + self.battle_state_length]
        x = x + self.battle_state_length
        player_active_pokemon_index = state[:, x].long()
        x = x + 1
        opponent_active_pokemon_index = state[:, x].long()

        # Get embeddings for each individual pokemon
        player_pokemon = []
        opponent_pokemon = []
        active_move1_batch = torch.zeros(
            state.shape[0], self.pokemon_embedding_dim, device=state.device
        )
        active_move2_batch = torch.zeros(
            state.shape[0], self.pokemon_embedding_dim, device=state.device
        )
        active_move3_batch = torch.zeros(
            state.shape[0], self.pokemon_embedding_dim, device=state.device
        )
        active_move4_batch = torch.zeros(
            state.shape[0], self.pokemon_embedding_dim, device=state.device
        )
        start = 0
        end = start + self.pokemon_state_length
        for i in range(6):
            (
                pokemon,
                active_move1,
                active_move2,
                active_move3,
                active_move4,
            ) = self.pokemon_model(
                player_pokemon_state[:, start:end], return_moveset=True
            )
            player_pokemon.append(pokemon)

            # Keep track of the active moves
            active_move1_batch[player_active_pokemon_index == i] = active_move1[
                player_active_pokemon_index == i
            ]
            active_move2_batch[player_active_pokemon_index == i] = active_move2[
                player_active_pokemon_index == i
            ]
            active_move3_batch[player_active_pokemon_index == i] = active_move3[
                player_active_pokemon_index == i
            ]
            active_move4_batch[player_active_pokemon_index == i] = active_move4[
                player_active_pokemon_index == i
            ]

            pokemon = self.pokemon_model(opponent_pokemon_state[:, start:end])
            opponent_pokemon.append(pokemon)

            start = end
            end = start + self.pokemon_state_length

        player_team = self.team_model(
            player_pokemon, player_state, player_active_pokemon_index
        )
        opponent_team = self.team_model(
            opponent_pokemon, opponent_state, opponent_active_pokemon_index
        )

        battle_state = torch.cat([player_team, opponent_team, battle_state], dim=-1)
        policy = self.policy_head(battle_state)
        value = self.value_head(battle_state)
        # 22 Actions:
        # 0-4, 4-8, 8-12, 12-16 are for moves.
        # Moves, Z-Moves, Mega Evolved Moves, Dynamaxed Moves
        # 16-22 = 6 Switches
        # TODO: Multiply/Add move vector by the relevant bonus (z/mega/dyna)
        emphasis_vector = torch.stack(
            [
                active_move1,
                active_move2,
                active_move3,
                active_move4,
                active_move1,
                active_move2,
                active_move3,
                active_move4,
                active_move1,
                active_move2,
                active_move3,
                active_move4,
                active_move1,
                active_move2,
                active_move3,
                active_move4,
                player_pokemon[0],
                player_pokemon[1],
                player_pokemon[2],
                player_pokemon[3],
                player_pokemon[4],
                player_pokemon[5],
            ],
            dim=1,
        )
        policy = torch.cat([policy.unsqueeze(-1), emphasis_vector], dim=-1)
        policy = self.policy_emphasis_head(policy).squeeze(-1)
        return policy, value


class FlattenedBattleModel(nn.Module):
    def __init__(
        self,
        n_obs,
        n_actions,
    ):
        super(FlattenedBattleModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_obs, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, 256),
            nn.ELU(inplace=True),
            nn.Linear(256, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, 64),
            nn.ELU(inplace=True),
            nn.Linear(64, n_actions),
        )

    def forward(self, state):
        return self.model(state)


class ActorCriticFlattenedBattleModel(nn.Module):
    def __init__(
        self,
        n_obs,
        n_actions,
    ):
        super(ActorCriticFlattenedBattleModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_obs, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, 256),
            nn.ELU(inplace=True),
            nn.Linear(256, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, 64),
            nn.ELU(inplace=True),
        )
        self.policy_head = nn.Linear(64, n_actions)
        self.value_head = nn.Linear(64, 1)

    def forward(self, state):
        features = self.model(state)
        policy = self.policy_head(features)
        value = self.value_head(features)
        return policy, value
