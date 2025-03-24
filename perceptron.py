import numpy as np
import matplotlib.pyplot as plt

def perceptron(operaciones, valores, entradas, salida, pesos, umbral, tasa_aprendizaje):
    epocas = 0
    max_epocas = 10000

    fig, ax = plt.subplots()
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    
    def plot():
        ax.clear()
        ax.axhline(0, color='black', lw=1)
        ax.axvline(0, color='black', lw=1)
        
        # Dibuja la recta de decisión
        if pesos[0] != 0 and pesos[1] != 0:
            x = np.linspace(-1.5, 1.5, 100)
            y = -(pesos[0] * x / pesos[1]) + (umbral / pesos[1])
            ax.plot(x, y, color='green', label='Recta de decisión')
        
        # Dibuja los puntos
        for s in range(operaciones):
            x, y = entradas[s]
            color = 'red' if salida[s] == 1 else 'blue'
            ax.scatter(x, y, color=color, s=40)
        
        ax.set_title(f'Épocas: {epocas}, Umbral: {umbral:.4f}')
        plt.pause(0.01)
    
    while epocas < max_epocas:
        plot()
        error = 0

        for s in range(operaciones):
            resultado = np.dot(entradas[s], pesos)
            resultado = 1 if resultado > umbral else 0

            if resultado != salida[s]:
                error = 1
                ajuste = salida[s] - resultado
                umbral -= ajuste * tasa_aprendizaje
                pesos += ajuste * entradas[s] * tasa_aprendizaje
        
        if error == 0:
            print("\nPerceptrón entrenado correctamente.")
            break

        epocas += 1
    plt.show()

def main():
    print("\na) Puerta AND\nb) Puerta OR\nc) Puerta NAND\nd) Puerta NOR\ne) Puerta XOR\nf) Datos normalizados")
    seleccion = input("Selecciona una opción: ").lower()
    
    if seleccion == 'f':
        entradas = np.array([
            [0.1, 0.2],   # Hombre
            [0.2, 0.3],   # Hombre
            [0.3, 0.4],   # Hombre
            [0.4, 0.5],   # Hombre
            [0.6, 0.7],   # Mujer
            [0.7, 0.8],   # Mujer
            [0.8, 0.9],   # Mujer
            [0.9, 1.0],   # Mujer
            [0.5, 0.6],   # Mujer
            [0.1, 0.15],  # Hombre
            [0.25, 0.35]  # Hombre
        ])
        
        salida = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0])
        pesos = np.random.rand(2)
        umbral = np.random.rand()
        tasa_aprendizaje = 0.1
        perceptron(len(entradas), 2, entradas, salida, pesos, umbral, tasa_aprendizaje)
    
    elif seleccion in ['a', 'b', 'c', 'd', 'e']:
        entradas = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])
        
        if seleccion == 'a':
            salida = np.array([0, 0, 0, 1])  # AND
        elif seleccion == 'b':
            salida = np.array([0, 1, 1, 1])  # OR
        elif seleccion == 'c':
            salida = np.array([1, 1, 1, 0])  # NAND
        elif seleccion == 'd':
            salida = np.array([1, 0, 0, 0])  # NOR
        elif seleccion == 'e':
            salida = np.array([0, 1, 1, 0])  # XOR
        
        pesos = np.random.rand(2)
        umbral = np.random.rand()
        tasa_aprendizaje = float(input("Introduce la tasa de aprendizaje: "))
        perceptron(len(entradas), 2, entradas, salida, pesos, umbral, tasa_aprendizaje)
    else:
        print("Opción no válida.")

if __name__ == "__main__":
    main()

