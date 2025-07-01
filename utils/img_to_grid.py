from PIL import Image
import numpy as np

TILE_SIZE = 8 

# Diccionario con algunos colores aproximados (puedes ajustarlos)
COLOR_MAP = {
    (228, 92, 16): '#',   # Muro naranja (puede ser variable)
    (0, 0, 0): ' ',        # Fondo
    (210, 164, 74): '.',   # Pellet
    (200, 72, 72): 'G',    # Fantasma rojo
    (198, 89, 179): 'G',   # Fantasma rosa
    (84, 138, 210): 'G',   # Fantasma azul
    (255, 255, 0): 'P'     # Pacman (amarillo)
}

def round_rgb(rgb, step=10):
    return tuple((int(c / step) * step for c in rgb))

def rgb_to_char(rgb, color_map, tolerance=30):
    """Encuentra el símbolo más cercano por color."""
    for ref_color, symbol in color_map.items():
        if all(abs(c1 - c2) < tolerance for c1, c2 in zip(rgb, ref_color)):
            return symbol
    return '?'  # Desconocido

def obs_to_grid(obs, tile_size=8):
    """Convierte una observación de MsPacman a una grilla de símbolos."""
    image = Image.fromarray(obs)
    width, height = image.size
    grid = []

    for y in range(0, height, tile_size):
        row = []
        for x in range(0, width, tile_size):
            tile = image.crop((x, y, x + tile_size, y + tile_size))
            tile_array = np.array(tile)
            avg_color = tuple(np.mean(tile_array.reshape(-1, 3), axis=0).astype(int))
            print(f"Tile at ({x}, {y}) - Avg Color: {avg_color}")  # Debugging line
            rounded_color = round_rgb(avg_color, step=10)
            symbol = rgb_to_char(rounded_color, COLOR_MAP)  
            row.append(symbol)
        grid.append(row)
    
    return grid


def main():
    # Cargar la imagen de ejemplo (puedes reemplazar esto con tu propia imagen)
    obs = np.array(Image.open("frame.png"))

    # Convertir la observación a una grilla
    grid = obs_to_grid(obs, TILE_SIZE)

    # Imprimir la grilla
    for row in grid:
        print(' '.join(row))

if __name__ == "__main__":
    main()