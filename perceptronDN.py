import numpy as np
import matplotlib.pyplot as plt

# Función de activación escalonada
def activacion_escalonada(valor, umbral):
    return 1 if valor > umbral else 0

# Entrenamiento del perceptrón
def perceptron(operaciones, valores, entradas, salida, pesos, umbral):
    epocas = 0
    max_epocas = 10000
    escala = 50
    
    # Configuración de la gráfica
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

    # Entrenamiento
    while epocas < max_epocas:
        plot()
        error = 0

        for s in range(operaciones):
            resultado = sum(entradas[s][i] * pesos[i] for i in range(valores))
            resultado = activacion_escalonada(resultado, umbral)

            if resultado != salida[s]:
                error = 1
                ajuste = salida[s] - resultado
                umbral -= ajuste

                for i in range(valores):
                    pesos[i] += ajuste * entradas[s][i]

        if error == 0:
            print("\nPerceptrón entrenado correctamente.")
            break

        epocas += 1

    # Mantener el gráfico abierto
    input("Presiona Enter para cerrar...")

# Función principal
def main():
    # Datos normalizados de 5 personas (Sexo, Edad, Peso, Altura, IMC)
    entradas = np.array([
        [0, 0.1, 0.25, 0.5, 0.18],  # Persona 1
        [1, 0.2, 0.3, 0.6, 0.22],   # Persona 2
        [0, 0.15, 0.35, 0.55, 0.19], # Persona 3
        [1, 0.3, 0.4, 0.7, 0.25],   # Persona 4
        [0, 0.4, 0.45, 0.75, 0.3],  # Persona 5
    ])

    # Salidas esperadas (0 o 1)
    salida = np.array([0, 0, 1, 1, 1])  # Clasificados en dos grupos linealmente separables

    # Inicialización de pesos y umbral
    pesos = np.zeros(5, dtype=float)  # 5 atributos por persona
    umbral = 0.5

    # Llamamos a la función de entrenamiento
    perceptron(5, 5, entradas, salida, pesos, umbral)

if __name__ == "__main__":
    main()
