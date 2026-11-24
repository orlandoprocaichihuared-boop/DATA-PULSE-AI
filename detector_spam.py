
# =====================================================
# PROYECTO DE MACHINE LEARNING
# DETECTOR DE MENSAJES SPAM
# Algoritmo: NAIVE BAYES
# Lenguaje: PYTHON
# =====================================================

# IMPORTAR LAS LIBRERIAS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# TITULO DEL PROGRAMA
print("=" * 55)
print("          PROYECTO DE MACHINE LEARNING")
print("             DETECTOR DE SPAM")
print("=" * 55)

# =====================================================
# 1. DATOS DE ENTRENAMIENTO
# =====================================================

# Estos son los ejemplos que utilizara el modelo.
# Cada mensaje tiene una categoria:
# 0 = MENSAJE NORMAL
# 1 = MENSAJE SPAM

mensajes = [
    "Hola como estas espero que estes bien",
    "Nos vemos mañana en la universidad",
    "La clase comienza a las ocho",
    "Ya llegue a casa",
    "Gracias por ayudarme con la tarea",
    "Puedes enviarme el documento",
    "Vamos a salir este fin de semana",
    "Te llamo cuando llegue",
    "Hoy tenemos reunion de trabajo",
    "Feliz cumpleaños que tengas un buen dia",
    "Estoy estudiando para el examen",
    "El profesor envio las diapositivas",
    "Nos encontramos en la biblioteca",
    "Muchas gracias por tu mensaje",
    "Ya termine el proyecto de clase",

    "Gana dinero facil desde tu casa",
    "Has ganado un premio reclama ahora",
    "Oferta exclusiva compra ahora",
    "Felicidades ganaste un celular gratis",
    "Haz clic aqui para reclamar tu premio",
    "Dinero gratis disponible para ti",
    "Ultima oportunidad gana miles de soles",
    "Reclama tu bono gratuito ahora",
    "Has sido seleccionado para ganar dinero",
    "Promocion increible compra ya",
    "Gana un premio sin hacer nada",
    "Oferta limitada ingresa aqui",
    "Tu premio te esta esperando",
    "Recibe dinero gratis hoy mismo",
    "Compra ahora y gana un regalo"
]

# Clasificacion de cada mensaje
# Los primeros 15 son normales (0)
# Los siguientes 15 son spam (1)

categorias = [
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1
]

# =====================================================
# 2. CREAR EL MODELO
# =====================================================

print("\nPreparando el modelo de Machine Learning...")

modelo = Pipeline([
    ("vectorizador", CountVectorizer(
        strip_accents="unicode",
        ngram_range=(1, 2)
    )),
    ("clasificador", MultinomialNB())
])

# =====================================================
# 3. ENTRENAR EL MODELO
# =====================================================

print("Entrenando con los ejemplos...")

modelo.fit(mensajes, categorias)

print("\n¡MODELO ENTRENADO CORRECTAMENTE!")
print("Cantidad de ejemplos:", len(mensajes))

# =====================================================
# 4. PROBAR EL MODELO CON MENSAJES NUEVOS
# =====================================================

print("\n" + "=" * 55)
print("          SISTEMA DE DETECCION DE SPAM")
print("=" * 55)

print("\nEscribe un mensaje para analizarlo.")
print("Escribe SALIR para cerrar el programa.")

while True:

    texto = input("\nEscribe tu mensaje: ").strip()

    # Comprobar si el usuario quiere salir
    if texto.lower() == "salir":
        print("\nGracias por utilizar el programa.")
        break

    # Comprobar que no este vacio
    if texto == "":
        print("Debes escribir un mensaje.")
        continue

    # El modelo analiza el mensaje
    resultado = modelo.predict([texto])[0]

    # Obtener la estimacion de probabilidad
    probabilidades = modelo.predict_proba([texto])[0]
    confianza = probabilidades[resultado] * 100

    # Mostrar el resultado
    print("\n========== RESULTADO ==========")

    if resultado == 1:
        print("ALERTA: POSIBLE MENSAJE SPAM")
    else:
        print("MENSAJE CLASIFICADO COMO NORMAL")

    print(f"Estimacion del modelo: {confianza:.2f}%")
    print("===============================")
