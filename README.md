# Optimización de PINNs para la Ecuación de Klein‑Gordon

Este repositorio contiene el código y los resultados del proyecto de investigación enfocado en la **optimización de Redes Neuronales Informadas por la Física (PINNs)** aplicadas a la ecuación de Klein‑Gordon homogénea en 1+1 dimensiones.  
El trabajo analiza los desafíos de convergencia de la función de coste, la rigidez de los gradientes y el uso de estrategias de regularización para alcanzar soluciones precisas.

---

## 📌 Descripción del problema

Se busca aproximar la solución de la ecuación de Klein‑Gordon:

\[
\frac{1}{c^2}\frac{\partial^2 \phi}{\partial t^2} - \nabla^2 \phi + \left(\frac{mc}{\hbar}\right)^2 \phi = 0,
\]

con condiciones iniciales de pulso gaussiano (partícula estática) y condiciones de contorno periódicas.  
El objetivo es entrenar una PINN que reproduzca la evolución temporal del campo escalar y compararla con una solución numérica de referencia obtenida mediante un método pseudoespectral.

---

## 🧠 Estrategia de optimización

La optimización de la PINN se formula como un problema multi‑objetivo con restricciones físicas. Se exploran dos enfoques:

1. **Restricción dura** → minimización de la pérdida de datos sujeta a que el residuo físico sea cero (KKT).  
   Se demuestra que el problema no es convexo, y el cálculo del Hessiano es intratable para arquitecturas profundas.

2. **Restricción blanda (regularización)** → se minimiza una función de coste total:

   \[
   \mathcal{L}(\theta) = \lambda_{ic}\mathcal{L}_{ic} + \lambda_{bc}\mathcal{L}_{bc} + \lambda_f\mathcal{L}_f,
   \]

   donde \(\mathcal{L}_f\) es el residuo de la EDP, y \(\lambda_i\) son pesos que equilibran la magnitud de los gradientes.

Para evitar el desequilibrio de gradientes (rigidez del término físico), se implementa el algoritmo **ADAM + Learning Rate Annealing (LRA)** propuesto por Wang et al. (2021), seguido de un refinamiento con **L‑BFGS**.

---

## ⚙️ Algoritmos de optimización

### 🔹 ADAM + LRA
- Actualiza los pesos \(\lambda_i\) cada cierto número de épocas usando la media móvil del cociente entre el máximo gradiente de la física y el promedio del gradiente de cada dato.
- Permite explorar la región de Pareto y estabilizar el entrenamiento.
- Se usaron 100 000 épocas con tasa de aprendizaje inicial \(10^{-4}\) y decaimiento exponencial.

### 🔹 L‑BFGS
- Optimizador de segundo orden (cuasi‑Newton) que refina la solución obtenida por ADAM.
- Converge en menos de 500 iteraciones, alcanzando pérdidas totales del orden de \(10^{-8}\).

---

## 📊 Resultados principales

- **ADAM + LRA** reduce la pérdida total a \(\sim 10^{-5}\), con pesos finales \(\lambda = [1.0,\ 4.90,\ 0.82]\).
- **L‑BFGS** logra una pérdida total de \(\sim 10^{-8}\), con una rápida convergencia de la pérdida física.
- La PINN reproduce cualitativamente el cono de luz y la dispersión de la onda, pero presenta un **error relativo \(L^2\) del 9.53 %** respecto a la simulación numérica, lo que indica que aún hay margen de mejora en la representación de amplitudes.

---

## 📁 Estructura del repositorio
