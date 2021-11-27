import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Flatten, Embedding, Add, Flatten
from tensorflow.keras.layers import Concatenate, Average, Softmax
from tensorflow.keras.models import Sequential, Model
import sys


def SimpleModel(n_action):
    """
    Our embeddings have shape (1, 10), which affects our hidden layer
    dimension and output dimension
    Flattening resolves potential issues that would arise otherwise
    """
    model = Sequential()
    model.add(Dense(128, activation="elu", input_shape=(1, 10)))
    model.add(Flatten())
    model.add(Dense(64, activation="elu"))
    model.add(Dense(n_action, activation="linear"))
    return model


def PokemonModel(pokemon_sample, embedding_dim, max_values):
    #### Pokemon Model ####
    # Define Inputs
    pokemon_species = Input(shape=(1,), name="pokemon_species")
    pokemon_item = Input(shape=(1,), name="pokemon_item")
    pokemon_ability = Input(shape=(1,), name="pokemon_ability")
    pokemon_possible_ability1 = Input(shape=(1,), name="pokemon_possible_ability1")
    pokemon_possible_ability2 = Input(shape=(1,), name="pokemon_possible_ability2")
    pokemon_possible_ability3 = Input(shape=(1,), name="pokemon_possible_ability3")
    pokemon_move1 = Input(shape=(1,), name="pokemon_move1")
    pokemon_move2 = Input(shape=(1,), name="pokemon_move2")
    pokemon_move3 = Input(shape=(1,), name="pokemon_move3")
    pokemon_move4 = Input(shape=(1,), name="pokemon_move4")
    pokemon_other = Input(shape=pokemon_sample[-1].shape, name="pokemon_other")
    # Define Embedding Layers
    species_embedding = Embedding(
        input_dim=max_values["species"],
        output_dim=embedding_dim,
        name="species_embedding",
    )
    item_embedding = Embedding(
        input_dim=max_values["items"], output_dim=embedding_dim, name="item_embedding"
    )
    ability_embedding = Embedding(
        input_dim=max_values["abilities"],
        output_dim=embedding_dim,
        name="ability_embedding",
    )
    move_embedding = Embedding(
        input_dim=max_values["moves"], output_dim=embedding_dim, name="move_embedding"
    )
    # Get Embeddings
    species_embedded = Flatten()(species_embedding(pokemon_species))
    item_embedded = Flatten()(item_embedding(pokemon_item))
    ability_embedded = Flatten()(ability_embedding(pokemon_ability))
    # Average of Possible Abilities
    possible_ability1_embedded = ability_embedding(pokemon_possible_ability1)
    possible_ability2_embedded = ability_embedding(pokemon_possible_ability2)
    possible_ability3_embedded = ability_embedding(pokemon_possible_ability3)
    possible_abilities_embedded = Average()(
        [
            possible_ability1_embedded,
            possible_ability2_embedded,
            possible_ability3_embedded,
        ]
    )
    possible_abilities_embedded = Flatten()(possible_abilities_embedded)
    # Average Embedding of 4 Moves
    move1_embedded = move_embedding(pokemon_move1)
    move2_embedded = move_embedding(pokemon_move2)
    move3_embedded = move_embedding(pokemon_move3)
    move4_embedded = move_embedding(pokemon_move4)
    moves_embedded = Average()(
        [move1_embedded, move2_embedded, move3_embedded, move4_embedded]
    )
    moves_embedded = Flatten()(moves_embedded)
    # Concatenate all Vectors
    pokemon_embedded = Concatenate()(
        [
            species_embedded,
            item_embedded,
            ability_embedded,
            possible_abilities_embedded,
            moves_embedded,
            pokemon_other,
        ]
    )
    pokemon_processed = Dense(embedding_dim, activation="relu", name="pokemon_fc")(
        pokemon_embedded
    )
    pokemon_model = Model(
        inputs=[
            pokemon_species,
            pokemon_item,
            pokemon_ability,
            pokemon_possible_ability1,
            pokemon_possible_ability2,
            pokemon_possible_ability3,
            pokemon_move1,
            pokemon_move2,
            pokemon_move3,
            pokemon_move4,
            pokemon_other,
        ],
        outputs=pokemon_processed,
    )
    return pokemon_model


def TeamModel(pokemon_model, embedding_dim, team_sample):
    # Pokemon Team Embedding
    pokemon_inputs = []
    for i in range(6):
        single_pokemon = []
        for model_input in pokemon_model.inputs:
            name = f"pokemon_{i+1}_{model_input.name.split(':')[0]}"
            single_pokemon.append(Input(shape=model_input.shape[1:], name=name))
        pokemon_inputs.append(single_pokemon)
    pokemon = [pokemon_model(pokemon_input) for pokemon_input in pokemon_inputs]
    team_pokemon = Concatenate()(pokemon)
    team_pokemon_embedding = Dense(embedding_dim, activation="relu")(team_pokemon)
    # Active Pokemon's Embedding
    active_pokemon_embedding = pokemon[0]
    # Team (Side) Information/Conditions
    team_inputs = Input(shape=team_sample.shape, name="team_conditions")
    # Concatenate
    team_embedded = Concatenate()(
        [team_pokemon_embedding, active_pokemon_embedding, team_inputs]
    )
    # Create Model
    team_model = Model(inputs=[pokemon_inputs, team_inputs], outputs=[team_embedded])

    return team_model


def BattleModel(team_model, n_actions, battle_sample):
    # Player Team Embedding
    player_team_inputs = []
    for model_input in team_model.inputs:
        name = f"player_{model_input.name.split(':')[0]}"
        player_team_inputs.append(Input(shape=model_input.shape[1:], name=name))
    player_team = team_model(player_team_inputs)
    # Opponent Team Embedding
    opponent_team_inputs = []
    for model_input in team_model.inputs:
        name = f"opponent_{model_input.name.split(':')[0]}"
        opponent_team_inputs.append(Input(shape=model_input.shape[1:], name=name))
    opponent_team = team_model(opponent_team_inputs)
    # Battle Information (Input) - Weather etc.
    battle_inputs = Input(shape=battle_sample.shape, name="battle_conditions")
    # Concatenate
    battle_information = Concatenate()([player_team, opponent_team, battle_inputs])
    # Battle Embedding - Get the final embedding of the entire battle
    battle_embedding = Dense(n_actions)(battle_information)
    # Create Mask for Invalid Actions
    mask = Input(shape=((n_actions)), name="action_mask")
    # Apply Mask 
    masked_outputs = Add()([battle_embedding, mask])
    # Softmax
    predictions = Softmax()(masked_outputs)
    # Create Model
    battle_model = Model(
        inputs=[player_team_inputs, opponent_team_inputs, battle_inputs, mask],
        outputs=predictions,
    )

    return battle_model


def FullStateModel(n_actions, state, embedding_dim, max_values):
    # State:
    # [*player_pokemon1, *player_pokemon2, *player_pokemon3, *player_pokemon4, 
    # *player_pokemon5, *player_pokemon6, player_state, *opponent_pokemon1, 
    # *opponent_pokemon2, *opponent_pokemon3, *opponent_pokemon4, 
    # *opponent_pokemon5, *opponent_pokemon6, opponent_state, battle_state, 
    # action_mask]
    # Where, each pokemon consists of 11 vectors:
    # pokemon_species, pokemon_item, pokemon_ability, 
    # pokemon_possible_ability1, pokemon_possible_ability2, 
    # pokemon_possible_ability3, pokemon_move1, pokemon_move2, pokemon_move3, 
    # pokemon_move4, pokemon_others_state
    # This means we have a total of 12 pokemon * 11 = 132 
    # 132 pokemon states + 2 team states + 1 battle state + 1 mask = 136
    
    # Here 0:11 = One Full Pokemon
    pokemon_model = PokemonModel(state[0:11], embedding_dim, max_values)
    # Player State = 3rd Last Vector
    team_model = TeamModel(pokemon_model, embedding_dim, state[-3])
    # Battle State = 2nd Last Vector
    battle_model = BattleModel(team_model, n_actions, state[-2])
    return battle_model
