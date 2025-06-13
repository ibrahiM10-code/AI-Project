import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Datos de ejemplo: usuarios y productos
users = ["Alice", "Bob", "Charlie"]
products = ["Laptop", "Smartphone", "Tablet", "Headphones"]

# Matriz de preferencias de usuarios (filas: usuarios, columnas: productos)
# 1 = le gusta, 0 = no le gusta
user_preferences = np.array([
    [1, 1, 0, 0],  # Alice
    [0, 1, 1, 0],  # Bob
    [1, 0, 0, 1],  # Charlie
])

def recommend_products(user_index, top_n=2):
    """
    Recomienda productos a un usuario basado en similitud de coseno con otros usuarios.
    """
    # Calcular similitud de coseno entre usuarios
    similarities = cosine_similarity([user_preferences[user_index]], user_preferences)[0]
    # Ignorar la similitud consigo mismo
    similarities[user_index] = 0

    # Calcular puntuación ponderada para cada producto
    weighted_scores = np.dot(similarities, user_preferences)
    # No recomendar productos que el usuario ya ha visto
    already_liked = user_preferences[user_index]
    recommendations = weighted_scores * (1 - already_liked)

    # Obtener los índices de los productos recomendados
    recommended_indices = recommendations.argsort()[::-1][:top_n]
    recommended_products = [products[i] for i in recommended_indices]

    return recommended_products

# Ejemplo de uso
if __name__ == "__main__":
    user = 0  # Alice
    print(f"Recomendaciones para {users[user]}: {recommend_products(user)}")