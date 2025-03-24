import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def perceptron(operaciones, valores, entradas, salida, pesos, umbral):
    epocas = 0
    max_epocas = 10000
    escala = 50

    fig, ax = plt.subplots()
    ax.set_xlim(-300, 300)
    ax.set_ylim(-300, 300)
    ax.set_aspect('equal')
    
    def plot():
        ax.clear()

        # Dibuja el plano cartesiano
        ax.axhline(0, color='black', lw=1)
        ax.axvline(0, color='black', lw=1)

        # Líneas de escala
        for x in range(-300, 300, escala):
            ax.plot([x, x], [-5, 5], color='black', lw=0.5)
            ax.plot([-5, 5], [x, x], color='black', lw=0.5)

        # Dibuja la recta de decisión
        if pesos[0] != 0 and pesos[1] != 0:
            x = np.linspace(-200, 200, 400)
            y = -(pesos[0] * x / pesos[1]) + (umbral / pesos[1]) * escala
            ax.plot(x, y, color='green', label='Recta de decisión')

        # Dibuja los puntos
        for s in range(operaciones):
            x = entradas[s][0] * escala
            y = -entradas[s][1] * escala
            color = 'red' if salida[s] == 1 else 'blue'
            ax.scatter(x, y, color=color, s=40)

        ax.set_title(f'Épocas: {epocas}, Umbral: {umbral}')
        plt.pause(0.01)

    while epocas < max_epocas:
        plot()
        error = 0

        for s in range(operaciones):
            resultado = sum(entradas[s][i] * pesos[i] for i in range(valores))

            if resultado > umbral:
                resultado = 1
            else:
                resultado = 0

            if resultado != salida[s]:
                error = 1
                ajuste = salida[s] - resultado
                umbral -= ajuste

                for i in range(valores):
                    pesos[i] += ajuste * entradas[s][i]

        print(f"\nÉpoca: {epocas}")
        print(f"Umbral: {umbral}")
        print("Pesos:", pesos)

        if error == 0:
            print("\nPerceptrón entrenado correctamente.")
            break

        epocas += 1

    plt.show()

def main():
    entradas = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    salida = np.zeros(4, dtype=int)
    
    print("\na) Puerta AND\nb) Puerta OR\nc) Puerta NAND \nd) Puerta NOR\ne) Puerta XOR")
    seleccion = input("Selecciona una opción: ").lower()

    if seleccion == 'a':
        salida = [0, 0, 0, 1]
    elif seleccion == 'b':
        salida = [0, 1, 1, 1]
    elif seleccion == 'c':
        salida = [1, 1, 1, 0]
    elif seleccion == 'd':
        salida = [1, 0, 0, 0]
    elif seleccion == 'e':
        salida = [0, 1, 1, 0]
    else:
        print("Opción no válida.")
        return

    pesos = np.zeros(2, dtype=int)
    umbral = 0

    pesos[0] = int(input("Introduce el peso 1: "))
    pesos[1] = int(input("Introduce el peso 2: "))
    umbral = int(input("Introduce el valor umbral: "))

    perceptron(4, 2, entradas, salida, pesos, umbral)

if __name__ == "__main__":
    main()
