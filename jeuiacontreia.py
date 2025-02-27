import pygame
import random
import ia
import time

# Initialisation de pygame
pygame.init()

# Dimensions de la fenêtre et de la carte
tile_size = 30
size = 20
width, height = size * tile_size, size * tile_size
interface_height = 150  # Hauteur supplémentaire pour l'interface

# Couleurs
PASSABLE_COLOR = (200, 200, 200)  # Gris clair pour les cases passables
PLAYER_COLOR = (0, 0, 255)  # Bleu pour le joueur
PLAYER_COLOR_LIGHT = (100, 100, 255)  # Bleu clair pour le joueur capable de bouger
ENEMY_COLOR = (255, 0, 0)  # Rouge pour les ennemis
ENEMY_COLOR_LIGHT = (255, 100, 100)  # Rouge clair pour les ennemis capables de bouger
SELECTED_COLOR = (0, 255, 0)  # Vert pour la sélection
OBJECTIVE_MAJOR_COLOR = (255, 255, 0)  # Jaune pour objectif majeur
OBJECTIVE_MINOR_COLOR = (255, 215, 0)  # Doré pour objectif mineur


# Classe pour les unités
class Unit:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.selected = False
        self.moved = False  # Indicateur de mouvement pour le tour
        self.pv = 2  # Points de Vie
        self.attacked_this_turn = False  # Indicateur d'attaque dans ce tour
        self.stunned = False  # Indicateur d'étourdissement pour le tour suivant

    def draw(self, screen, units, objectives):
        """Affiche l'unité sur l'écran."""
        rect = pygame.Rect(self.x * tile_size, self.y * tile_size, tile_size, tile_size)
        if not self.moved:
            color = (
                PLAYER_COLOR_LIGHT if self.color == PLAYER_COLOR else ENEMY_COLOR_LIGHT
            )
        else:
            color = self.color
        pygame.draw.rect(screen, color, rect)

        if self.selected:
            pygame.draw.rect(screen, SELECTED_COLOR, rect, 3)

        font = pygame.font.SysFont(None, 16)
        symbols = self.get_symbols_on_same_tile(units)
        combined_text = font.render(symbols, True, (255, 255, 255))
        text_width = combined_text.get_width()
        text_x = self.x * tile_size + (tile_size - text_width) // 2
        screen.blit(combined_text, (text_x, self.y * tile_size + 5))

        for obj in objectives:
            if self.x == obj["x"] and self.y == obj["y"]:
                pygame.draw.rect(screen, (0, 255, 0), rect, 1)

    def can_move(self, x, y, units):
        """Vérifie si l'unité peut se déplacer vers une case."""
        if 0 <= x < size and 0 <= y < size:
            if abs(self.x - x) <= 1 and abs(self.y - y) <= 1:
                return True
        return False

    def move(self, x, y, units):
        """Déplace l'unité vers une case spécifiée."""
        if self.can_move(x, y, units):
            self.x = x
            self.y = y
            self.moved = True

    def attack(self, target_unit, units, objectives):
        """Attaque une unité ennemie."""
        if self.can_move(target_unit.x, target_unit.y, units):
            dx = target_unit.x - self.x
            dy = target_unit.y - self.y
            new_x, new_y = target_unit.x + dx, target_unit.y + dy

            if target_unit.attacked_this_turn:
                target_unit.pv -= 1
                target_unit.attacked_this_turn = False
                if target_unit.pv <= 0:
                    units.remove(target_unit)
                    return
                elif 0 <= new_x < size and 0 <= new_y < size:
                    if any(u.x == new_x and u.y == new_y and u.color != target_unit.color for u in units) or any(obj['x'] == new_x and obj['y'] == new_y and obj['type'] == 'MAJOR' for obj in objectives):
                        units.remove(target_unit)
                    else:
                        target_unit.move(new_x, new_y, units)
            else:
                if 0 <= new_x < size and 0 <= new_y < size:
                    if any(u.x == new_x and u.y == new_y and u.color != target_unit.color for u in units) or any(obj['x'] == new_x and obj['y'] == new_y for obj in objectives):
                        units.remove(target_unit)
                    else:
                        target_unit.move(new_x, new_y, units)
                        target_unit.attacked_this_turn = True

    def get_symbols_on_same_tile(self, units):
        """Retourne les symboles des unités sur la même case."""
        symbols = [u.get_symbol() for u in units if u.x == self.x and u.y == self.y]
        return " ".join(symbols)

    def get_symbol(self):
        """Retourne le symbole de l'unité."""
        return "U"


# Générer la carte
def generate_map(size):
    """Génère une carte de taille spécifiée."""
    return [[1 for _ in range(size)] for _ in range(size)]


# Afficher la carte
def draw_map(screen, game_map, tile_size):
    """Affiche la carte."""
    for y in range(len(game_map)):
        for x in range(len(game_map[y])):
            color = PASSABLE_COLOR
            pygame.draw.rect(
                screen, color, (x * tile_size, y * tile_size, tile_size, tile_size)
            )


# Générer des unités sur des cases passables uniquement
def generate_units():
    """Génère les unités pour les joueurs et les ennemis."""
    units = []
    player_positions = [(0, i) for i in range(size)]
    enemy_positions = [(size - 1, i) for i in range(size)]

    player_positions = random.sample(player_positions, 7)
    enemy_positions = random.sample(enemy_positions, 7)

    player_units = [Unit(*pos, PLAYER_COLOR) for pos in player_positions]
    enemy_units = [Unit(*pos, ENEMY_COLOR) for pos in enemy_positions]

    units.extend(player_units)
    units.extend(enemy_units)

    return units


# Ajouter des objectifs à la carte
def add_objectives():
    """Ajoute des objectifs à la carte."""
    objectives = []
    center_x, center_y = size // 2, size // 2
    while True:
        x, y = random.randint(center_x - 3, center_x + 3), random.randint(
            center_y - 3, center_y + 3
        )
        if not any(obj["x"] == x and obj["y"] == y for obj in objectives):
            objectives.append({"x": x, "y": y, "type": "MAJOR"})
            break

    for _ in range(3):
        while True:
            x, y = random.randint(center_x - 5, center_x + 5), random.randint(
                center_y - 5, center_y + 5
            )
            if not any(obj["x"] == x and obj["y"] == y for obj in objectives):
                objectives.append({"x": x, "y": y, "type": "MINOR"})
                break

    return objectives


# Afficher les objectifs
def draw_objectives(screen, objectives, tile_size):
    """Affiche les objectifs sur la carte."""
    for obj in objectives:
        color = (
            OBJECTIVE_MAJOR_COLOR if obj["type"] == "MAJOR" else OBJECTIVE_MINOR_COLOR
        )
        pygame.draw.rect(
            screen,
            color,
            (obj["x"] * tile_size, obj["y"] * tile_size, tile_size, tile_size),
        )


# Calculer les scores
def calculate_scores(units, objectives):
    """Calcule les scores des joueurs et des ennemis en fonction des objectifs contrôlés."""
    player_score = 0
    enemy_score = 0

    for obj in objectives:
        if any(
            unit.x == obj["x"] and unit.y == obj["y"] and unit.color == PLAYER_COLOR
            for unit in units
        ):
            player_score += 3 if obj["type"] == "MAJOR" else 1
        elif any(
            unit.x == obj["x"] and unit.y == obj["y"] and unit.color == ENEMY_COLOR
            for unit in units
        ):
            enemy_score += 3 if obj["type"] == "MAJOR" else 1

    return player_score, enemy_score


# Afficher le message de changement de tour
def draw_turn_indicator(screen, player_turn):
    """Affiche l'indicateur de tour."""
    font = pygame.font.SysFont(None, 36)
    text = "Joueur" if player_turn else "Ennemi"
    img = font.render(text, True, (255, 255, 255))
    screen.blit(img, (10, 10))


# Afficher le bouton de changement de tour
def draw_end_turn_button(screen, width, height, interface_height):
    """Affiche le bouton de fin de tour."""
    font = pygame.font.SysFont(None, 36)
    text = font.render("Terminé", True, (255, 255, 255))
    button_rect = pygame.Rect(width // 2 - 50, height, 100, interface_height - 10)
    pygame.draw.rect(screen, (100, 100, 100), button_rect)
    screen.blit(text, (width // 2 - 50 + 10, height + 10))


# Vérifier si le bouton de changement de tour est cliqué
def end_turn_button_clicked(mouse_pos, width, height, interface_height):
    """Vérifie si le bouton de fin de tour a été cliqué."""
    x, y = mouse_pos
    button_rect = pygame.Rect(width // 2 - 50, height, 100, interface_height - 10)
    return button_rect.collidepoint(x, y)


# Afficher les attributs de l'unité sélectionnée
def draw_unit_attributes(screen, unit, width, height, interface_height):
    """Affiche les attributs de l'unité sélectionnée."""
    if unit:
        font = pygame.font.SysFont(None, 24)
        pv_text = f"PV: {unit.pv} / 2"
        unit_img = font.render("Unité", True, (255, 255, 255))
        pv_img = font.render(pv_text, True, (255, 255, 255))
        screen.blit(unit_img, (10, height + 10))
        screen.blit(pv_img, (10, height + 40))


# Afficher les scores
def draw_scores(screen, player_score, enemy_score, width, height):
    """Affiche les scores des joueurs."""
    font = pygame.font.SysFont(None, 24)
    font = pygame.font.SysFont(None, 24)
    player_score_text = f"Score Joueur: {player_score}"
    enemy_score_text = f"Score Ennemi: {enemy_score}"
    player_score_img = font.render(player_score_text, True, (255, 255, 255))
    enemy_score_img = font.render(enemy_score_text, True, (255, 255, 255))
    screen.blit(player_score_img, (10, height + 70))
    screen.blit(enemy_score_img, (width - 150, height + 70))

#compter le nombres de carrés (alliés et énemies)
def count_units(units):
    """Compte le nombre d'unités vivantes pour chaque équipe."""
    player_units = sum(1 for unit in units if unit.color == PLAYER_COLOR)
    enemy_units = sum(1 for unit in units if unit.color == ENEMY_COLOR)
    return player_units, enemy_units


#Affiche le nombre de joueurs encore vivants
def draw_life_bars(screen, player_life_bars, enemy_life_bars, width, height):
    """Affiche les barres de vie des joueurs."""
    bar_width = 20
    bar_height = 10
    spacing = 5
    x_offset = 10
    y_offset = height + 100

    for i in range(player_life_bars):
        pygame.draw.rect(screen, PLAYER_COLOR, (x_offset + i * (bar_width + spacing), y_offset, bar_width, bar_height))

    for i in range(enemy_life_bars):
        pygame.draw.rect(screen, ENEMY_COLOR, (width - x_offset - (i + 1) * (bar_width + spacing), y_offset, bar_width, bar_height))


# Afficher le message de victoire
def draw_victory_message(screen, message, width, height):
    """Affiche le message de victoire."""
    font = pygame.font.SysFont(None, 48)
    victory_img = font.render(message, True, (255, 255, 255))
    screen.blit(victory_img, (width // 2 - 100, height // 2 - 24))


def run_game(auto_play=False, display=False, epsilon = 0.1):
    # Configuration de la fenêtre
    if display:
        screen = pygame.display.set_mode((width, height + interface_height))
        pygame.display.set_caption("Carte de 20x20 avec unités et déplacement")

    # Générer une carte de 20 par 20
    game_map = generate_map(size)

    # Générer les unités
    units = generate_units()

    # Ajouter des objectifs
    objectives = add_objectives()

    selected_unit = None
    player_turn = True  # True pour le tour du joueur, False pour le tour de l'ennemi
    player_score = 0
    enemy_score = 0
    victory = False
    victory_message = ""

    # Boucle principale du jeu
    running = True
    while running:
        if not victory:
            if not auto_play:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                        player_turn = not player_turn
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        x, y = event.pos
                        if end_turn_button_clicked(
                            (x, y), width, height, interface_height
                        ):
                            player_turn = not player_turn
                        else:
                            grid_x, grid_y = x // tile_size, y // tile_size
                            if event.button == 1:  # Clic gauche pour sélectionner
                                possible_units = [
                                    u
                                    for u in units
                                    if u.x == grid_x
                                    and u.y == grid_y
                                    and not u.moved
                                    and u.color
                                    == (PLAYER_COLOR if player_turn else ENEMY_COLOR)
                                ]
                                if selected_unit in possible_units:
                                    current_index = possible_units.index(selected_unit)
                                    selected_unit.selected = False
                                    selected_unit = possible_units[
                                        (current_index + 1) % len(possible_units)
                                    ]
                                else:
                                    if selected_unit:
                                        selected_unit.selected = False
                                    if possible_units:
                                        selected_unit = possible_units[0]
                                if selected_unit:
                                    selected_unit.selected = True
                            elif (
                                event.button == 3
                            ):  # Clic droit pour déplacer ou attaquer
                                if selected_unit and selected_unit.color == (
                                    PLAYER_COLOR if player_turn else ENEMY_COLOR
                                ):
                                    target_unit = [
                                        u
                                        for u in units
                                        if u.x == grid_x
                                        and u.y == grid_y
                                        and u.color != selected_unit.color
                                    ]

                                    for cible in target_unit:
                                        selected_unit.attack(cible, units, objectives)

                                    if selected_unit.can_move(grid_x, grid_y, units):
                                        selected_unit.move(grid_x, grid_y, units)
                                        selected_unit.selected = False
                                        selected_unit = None
            else:
                if not player_turn:
                    ia.enemy_turn(units, objectives, size, epsilon = epsilon)
                    #time.sleep(1)
                else:
                    ia.player_turn(units, objectives, size, epsilon = epsilon)
                    #time.sleep(1)
                player_turn = not player_turn

            for unit in units:
                unit.moved = False  # Réinitialiser l'indicateur de mouvement
                unit.attacked_this_turn = False  # Réinitialiser l'indicateur d'attaque

            player_score_turn, enemy_score_turn = calculate_scores(units, objectives)
            player_score += player_score_turn
            enemy_score += enemy_score_turn

            if player_score >= 500:
                victory = True
                victory_message = "Victoire Joueur!"
            elif enemy_score >= 500:
                victory = True
                victory_message = "Victoire Ennemi!"
            elif not any(unit.color == PLAYER_COLOR for unit in units):
                victory = True
                victory_message = "Victoire Ennemi!"
            elif not any(unit.color == ENEMY_COLOR for unit in units):
                victory = True
                victory_message = "Victoire Joueur!"

            if display:
                screen.fill((0, 0, 0))
                draw_map(screen, game_map, tile_size)
                draw_objectives(screen, objectives, tile_size)

                for unit in units:
                    unit.draw(screen, units, objectives)

                draw_turn_indicator(screen, player_turn)
                draw_end_turn_button(screen, width, height, interface_height)
                draw_unit_attributes(
                    screen, selected_unit, width, height, interface_height
                )
                draw_scores(screen, player_score, enemy_score, width, height)

                #Compter le nombre de carrés encore vivants sur la carte
                player_units, enemy_units = count_units(units)
                draw_life_bars(screen, player_units, enemy_units, width, height)

                if victory:
                    draw_victory_message(screen, victory_message, width, height)
                    pygame.display.flip()
                    pygame.time.wait(3000)
                    running = False

                pygame.display.flip()

    return player_score, enemy_score


if __name__ == "__main__":
    run_game(auto_play=True, display=True, epsilon=0.1)
