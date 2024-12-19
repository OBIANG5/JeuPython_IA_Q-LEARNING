import random
import numpy as np
import pickle

# Hyperparamètres du Q-Learning
gamma = 0.95 # Facteur d'actualisation, voir sur le long terme (Discount Factor)
alpha = 0.1 # Vitesse d'apprentissage (Learning Rate)
epsilon = 0.1 # Taux d'exploration (Exploration Rate)

# Q-Table, qui stock les valeurs pour chaque état/action
q_table = {}

# Convertit l'état du jeu en une clé unique pour accéder à la Q-table
def state_to_key(state):
    return tuple((unit.x, unit.y, unit.color) for unit in state['units']) + \
    tuple((obj['x'], obj['y'], obj['type']) for obj in state['objectives'])

# Retourne la liste des actions possibles pour une unité donnée
def get_possible_actions(unit, units, size):
    actions = []
    for dx in [-1, 0, 1]:  # Explorer les déplacements dans les directions x (-1, 0, 1)
        for dy in [-1, 0, 1]:  # Explorer les déplacements dans les directions y (-1, 0, 1)
            new_x, new_y = unit.x + dx, unit.y + dy  # Calculer la nouvelle position
            # Vérifier si la nouvelle position est dans les limites et n'est pas occupée par une unité alliée
            if 0 <= new_x < size and 0 <= new_y < size and not any(u.x == new_x and u.y == new_y and u.color == unit.color for u in units):
                actions.append((new_x, new_y))  # Ajouter l'action à la liste des actions possibles
    return actions

# Initialise la Q-table pour un état donné si ce n'est pas déjà fait
def initialize_q_table(state, units, size):
    key = state_to_key(state)  # Obtenir la clé unique pour l'état actuel
    if key not in q_table:
        q_table[key] = {}
        for unit in state['units']:
            if unit.color in [(255, 0, 0), (0, 0, 255)]:  # Seules les unités ennemies ou alliées sont prises en compte
                for action in get_possible_actions(unit, units, size):  # Pour chaque action possible
                    if action not in q_table[key]:
                        q_table[key][action] = 0  # Initialiser la valeur Q à 0

# Met à jour la Q-table en utilisant l'équation du Q-learning
def update_q_table(state, action, reward, next_state, units, size):
    key = state_to_key(state)  # Clé de l'état actuel
    next_key = state_to_key(next_state)  # Clé de l'état suivant

    initialize_q_table(state, units, size)  # Assurer que la Q-table est initialisée pour l'état actuel
    initialize_q_table(next_state, units, size)  # Assurer que la Q-table est initialisée pour l'état suivant

    # Trouver la meilleure action possible dans l'état suivant
    best_next_action = max(q_table[next_key], key=q_table[next_key].get, default=(0, 0))

    # Calculer la cible TD (Temporal Difference target) : récompense + valeur estimée de la meilleure action dans l'état suivant
    td_target = reward + gamma * q_table[next_key].get(best_next_action, 0)

    # Calculer l'erreur TD : différence entre la cible TD et la valeur Q actuelle
    td_error = td_target - q_table[key].get(action, 0)

    # Mettre à jour la valeur Q pour l'action dans l'état actuel
    q_table[key][action] = q_table[key].get(action, 0) + alpha * td_error

# Calcule la récompense pour une unité en fonction de sa position et des objectifs
def get_reward(unit, objectives, units, size):
    reward = 0
    for obj in objectives:
        if unit.x == obj['x'] and unit.y == obj['y']:  # Si l'unité est sur un objectif
            reward += 3 if obj['type'] == 'MAJOR' else 1  # Récompense plus élevée pour un objectif majeur

    for target in units:
        # Récompenser l'attaque d'une unité ennemie à proximité
        if target.color != unit.color and abs(unit.x - target.x) <= 1 and abs(unit.y - target.y) <= 1:
            reward += 10

    # Pénalité pour être dans les coins de la carte (zones plus risquées)
    if (unit.x == 0 and unit.y == 0) or (unit.x == 0 and unit.y == size - 1) or \
       (unit.x == size - 1 and unit.y == 0) or (unit.x == size - 1 and unit.y == size - 1):
        reward -= 5

    return reward

# Exécute une action choisie par l'IA : se déplacer ou attaquer
def perform_action(unit, action, units, objectives, size):
    target_x, target_y = action  # Position cible de l'action
    target_unit = next((u for u in units if u.x == target_x and u.y == target_y and u.color != unit.color), None)  # Chercher une unité ennemie à la position cible

    if target_unit:  # Si une unité ennemie est présente, attaquer
        unit.attack(target_unit, units, objectives)
    else:  # Sinon, se déplacer
        unit.move(target_x, target_y, units)

# Choisit une action de manière stratégique en tenant compte des objectifs et des coins
def strategic_choose_action(state, unit, units, size, objectives, epsilon=epsilon):
    key = state_to_key(state)
    initialize_q_table(state, units, size)

    possible_actions = get_possible_actions(unit, units, size)  # Actions possibles
    if not possible_actions:
        return (unit.x, unit.y)  # Si aucune action possible, rester sur place

    if random.uniform(0, 1) < epsilon:  # Exploration
        return random.choice(possible_actions)
    else:
        action_values = {action: q_table[key].get(action, 0) for action in possible_actions}
    
        # Préférence pour les actions qui capturent des objectifs
        objective_actions = [action for action in possible_actions if any(obj['x'] == action[0] and obj['y'] == action[1] for obj in objectives)]
        if objective_actions:
            return random.choice(objective_actions)  # Choisir aléatoirement parmi les actions d'objectif
        
        # Préférence pour les actions qui attaquent des unités ennemies
        for action in possible_actions:
            target_unit = next((u for u in units if u.x == action[0] and u.y == action[1] and u.color != unit.color), None)
            if target_unit:
                return action
        
        # Éviter les coins de la carte
        non_corner_actions = [action for action in possible_actions if (action[0] != 0 and action[0] != size - 1 and action[1] != 0 and action[1] != size - 1)]
        if non_corner_actions:
            return random.choice(non_corner_actions)  # Choisir aléatoirement parmi les actions non-corners
        
        # Sinon, choisir l'action avec la meilleure valeur Q
        best_action = max(action_values, key=action_values.get)
        return best_action

# Gère le tour de l'IA ennemie
def enemy_turn(units, objectives, size, epsilon):
    state = {'units': units, 'objectives': objectives}
    all_units_moved = True  # Indicateur pour vérifier si toutes les unités ont bougé
    objective_units = 0  # Nombre d'unités ennemies sur les objectifs
    for unit in units:
        if unit.color == (255, 0, 0):  # ENEMY_COLOR
            initial_position = (unit.x, unit.y)  # Position initiale de l'unité
            action = strategic_choose_action(state, unit, units, size, objectives, epsilon)  # Choisir une action de manière stratégique
            perform_action(unit, action, units, objectives, size)  # Exécuter l'action
            next_state = {'units': units, 'objectives': objectives}
            reward = get_reward(unit, objectives, units, size)  # Calculer la récompense
            if (unit.x, unit.y) == initial_position:  # Si l'unité n'a pas bougé
                reward -= 1  # Pénalité pour ne pas avoir bougé
                all_units_moved = False
            update_q_table(state, action, reward, next_state, units, size)  # Mettre à jour la Q-table
            if any(obj['x'] == unit.x and obj['y'] == unit.y for obj in objectives):  # Vérifier si l'unité est sur un objectif
                objective_units += 1

    # S'assurer que deux unités ennemies occupent des objectifs
    if objective_units < 2:
        assigned_units = 0
        for unit in units:
            if unit.color == (255, 0, 0):  # ENEMY_COLOR
                if assigned_units >= 2:
                    break
                for obj in objectives:
                    if not any(u.x == obj['x'] and u.y == obj['y'] for u in units):  # Chercher un objectif inoccupé
                        action = (obj['x'], obj['y'])
                        perform_action(unit, action, units, objectives, size)
                        next_state = {'units': units, 'objectives': objectives}
                        reward = get_reward(unit, objectives, units, size)
                        update_q_table(state, action, reward, next_state, units, size)
                        assigned_units += 1
                        break

    # Pénalité globale si au moins une unité ne bouge pas
    if not all_units_moved:
        for unit in units:
            if unit.color == (255, 0, 0):
                state = {'units': units, 'objectives': objectives}
                reward = -5
                next_state = {'units': units, 'objectives': objectives}
                update_q_table(state, (unit.x, unit.y), reward, next_state, units, size)

def player_turn(units, objectives, size, epsilon):
    state = {'units': units, 'objectives': objectives}
    all_units_moved = True
    objective_units = 0
    for unit in units:
        if unit.color == (0, 0, 255):  # PLAYER_COLOR
            initial_position = (unit.x, unit.y)
            action = strategic_choose_action(state, unit, units, size, objectives, epsilon)
            perform_action(unit, action, units, objectives, size)
            next_state = {'units': units, 'objectives': objectives}
            reward = get_reward(unit, objectives, units, size)
            if (unit.x, unit.y) == initial_position:
                reward -= 1  # Pénalité si l'unité n'a pas bougé
                all_units_moved = False
            update_q_table(state, action, reward, next_state, units, size)
            if any(obj['x'] == unit.x and obj['y'] == unit.y for obj in objectives):
                objective_units += 1

    # Si moins de deux unités alliées sont sur les objectifs, assigner deux unités aux objectifs
    if objective_units < 2:
        assigned_units = 0
        for unit in units:
            if unit.color == (0, 0, 255):  # PLAYER_COLOR
                if assigned_units >= 2:
                    break
                for obj in objectives:
                    if not any(u.x == obj['x'] and u.y == obj['y'] for u in units):
                        action = (obj['x'], obj['y'])
                        perform_action(unit, action, units, objectives, size)
                        next_state = {'units': units, 'objectives': objectives}
                        reward = get_reward(unit, objectives, units, size)
                        update_q_table(state, action, reward, next_state, units, size)
                        assigned_units += 1
                        break

    if not all_units_moved:
        for unit in units:
            if unit.color == (0, 0, 255):
                state = {'units': units, 'objectives': objectives}
                reward = -5  # Pénalité globale si au moins une unité ne bouge pas
                next_state = {'units': units, 'objectives': objectives}
                update_q_table(state, (unit.x, unit.y), reward, next_state, units, size)

# Sauvegarder la Q-Table dans un fichier
def save_q_table(filename='q_table.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(q_table, f)

# Charger la Q-Table depuis le fichier
def load_q_table(filename='q_table.pkl'):
    global q_table
    with open(filename, 'rb') as f:
        q_table = pickle.load(f)
