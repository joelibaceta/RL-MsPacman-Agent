# Entrenando a Ms. Pac-Man con Aprendizaje por Refuerzo Profundo

Allá por los años 80, Pac-Man conquistó los arcades comiendo puntitos y escapando de fantasmas con más carisma que muchos protagonistas de películas. Pero luego llegó ella: una heroína con moño y más actitud, Ms. Pac-Man, la secuela que muchos tuvimos en el Atari 2600, donde los laberintos eran más impredecibles, los fantasmas más rebeldes, y la IA… inexistente.

Hoy, cuatro décadas después, queremos ver qué tan lejos puede llegar una inteligencia artificial moderna si le damos la tarea de dominar este clásico inmortal. ¿Puede una red neuronal convolucional aprender a comerse todos los puntos, esquivar a Pinky y cazar a Inky cuando se vuelve azul?

Este repositorio es el resultado de ese experimento: enseñamos a Ms. Pac-Man a jugar con refuerzos, sin reglas preprogramadas, sin ver el código del juego, solo a partir de las imágenes del entorno y las recompensas que recibe.

Como los humanos. Pero sin joystick.

---

## Pero ... ¿Cómo aprende una IA a jugar sola?

A diferencia de otras técnicas de inteligencia artificial donde le mostramos a la máquina ejemplos correctos (como “esto es un gato, esto no”), el Aprendizaje por Refuerzo (RL) funciona más como entrenar a una mascota.

En lugar de decirle qué hacer, le decimos si lo hizo bien o mal.

Imagina que tienes un robot en un laberinto, y le das puntos si encuentra una salida y le quitas puntos si choca contra una pared. Con el tiempo, el robot aprende que ciertas acciones llevan a mejores resultados, y comienza a desarrollar su propio plan.

En RL hablamos de:
- Agente: nuestra IA (el cerebro de Ms. Pac-Man).
- Entorno: el mundo que la IA observa (el laberinto del juego).
- Acciones: moverse arriba, abajo, izquierda o derecha.
- Recompensas: positivos por comer puntos, frutas o fantasmas, negativos por morir.
- Política: la estrategia que el agente aprende para decidir qué acción tomar en cada momento.

Con miles o millones de partidas, la IA va aprendiendo por prueba y error qué decisiones valen la pena. Como un niño que aprende a no tocar el fuego… después de algunas quemaduras.

Y lo más interesante: todo eso lo hace solo observando los píxeles del juego, sin saber reglas, sin trucos, sin memoria interna del mapa.

## ¿Y cómo hicimos que aprendiera? 

Entrenar a Ms. Pac-Man no es tan simple como soltar una IA y decirle “buena suerte”. Tuve que preparar cuidadosamente el entorno para que pueda ver, aprender, y mejorar sin instrucciones.

### ¿Qué necesitamos?**
1.	El juego: usamos la versión oficial de Ms. Pac-Man de Atari.
2.	Un entorno controlado: como en un experimento científico, necesitamos que cada partida sea medible, reproducible y clara.
3.	Una forma de ver: la IA no recibe texto ni comandos, solo imágenes RGB del juego (como si viera una pantalla).
4.	Un sistema de recompensas: puntos por comer, penalidades por morir.
5.	Una memoria: como el juego no tiene historia (cada imagen es un instante), le damos cuatro imágenes seguidas para que entienda el movimiento.
6.	Un cerebro artificial: entrenado con un algoritmo llamado DQN (Deep Q-Network), que combina redes neuronales y aprendizaje por refuerzo.

## Ahora algo de contexto técnico

### ¿Qué es Gymnasium?

Gymnasium es una librería que simula entornos clásicos de juegos y tareas (como caminar, jugar, correr, etc.) que pueden ser controlados por agentes de inteligencia artificial. Cada entorno actúa como un laboratorio virtual donde la IA puede observar lo que ocurre (una imagen o vector), tomar una acción (moverse, saltar, disparar), y recibir una recompensa en función de lo que hizo bien o mal.

En nuestro caso, usamos ALE/MsPacman-v5 la version de Ms Pacman adaptada para Gymnasium, que nos permite interactuar con el juego de manera controlada y medible.


###  ¿Qué es DQN?

DQN (Deep Q-Network) es un algoritmo que combina redes neuronales con Q-learning, una técnica de RL que aprende a predecir cuán buena será una acción en un estado determinado. Básicamente, es como enseñarle a un jugador novato a jugar mejor solo con ensayo y error, hasta que entienda que “comer fantasmas = bueno” y “chocar contra ellos = malo”.


El objetivo de Q-learning es aprender una función llamada **Q(s, a)**, que estima el **valor esperado** de tomar una acción `a` en un estado `s`, siguiendo una buena estrategia (política). En otras palabras: ¿qué tan buena es esta acción en este momento?

La **ecuación de Bellman** para Q-learning es:

\[
Q(s, a) = r + \gamma \max_{a'} Q(s', a')
\]

donde:

- \( s \): estado actual del juego (ej. la imagen de Pac-Man en un frame),
- \( a \): acción tomada (ej. moverse a la izquierda),
- \( r \): recompensa obtenida tras esa acción,
- \( s' \): nuevo estado alcanzado,
- \( \gamma \): factor de descuento (cuánto valen las recompensas futuras),
- \( a' \): acción siguiente considerada.

