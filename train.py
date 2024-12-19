from ia import save_q_table

def train(num_episodes=100, epsilon_decay=0.1, min_epsilon=0.01):
    # Importation ici pour eviter des erreurs
    from jeuiacontreia import run_game

    # Initialisation de la valeur epsilon (probabilité d'explorer plutôt que d'exploiter)
    epsilon = 1
    
    # Boucle principale d'entraînement
    for episode in range(num_episodes):
        print(f"Début de l'épisode {episode + 1}")
        
        # Exécution d'une partie
        player_score, enemy_score = run_game(auto_play=True, display=True)
        
        # Affichage des scores après chaque match
        print(f"Match {episode + 1}/{num_episodes} terminé. Score Joueur: {player_score}, Score Ennemi: {enemy_score}")
        
        # Réduction de la valeur epsilon à chaque match pour favoriser progressivement l'exploitation des stratégies apprises
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
    
    print("Entraînement terminé")
    
    # Sauvegarde de la Q-table (les valeurs Q apprises)
    save_q_table()

if __name__ == "__main__":
    train()