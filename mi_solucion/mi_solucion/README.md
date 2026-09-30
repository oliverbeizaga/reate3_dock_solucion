# Solución: Create 3 Dock Challenge
Este paquete de ROS 2 Humble resuelve el reto de acoplamiento autónomo usando el LiDAR del iRobot Create 3.

## Características Técnicas
* **FSM Robusto:** Implementa una Máquina de Estados (SEARCH, WANDER, GO_TO_PREDOCK, ALIGN, APPROACH, RECOVER, DOCKED).
* **Exploración Activa (Wander):** Si el robot inicia en una esquina ciega y no detecta el dock tras 15 segundos de giro, esquiva obstáculos y se reubica para volver a buscar.
* **Percepción Avanzada:** Usa `tf2_ros` para corregir la rotación física del LiDAR y aplica `RANSAC` para detectar la geometría exacta (cajas de 8cm y hueco de 9.5cm).
* **Control Lookahead:** Acercamiento milimétrico con corrección lateral dinámica.

## Ejecución (Regla de 1 comando)
```bash
ros2 launch mi_solucion solucion.launch.py
