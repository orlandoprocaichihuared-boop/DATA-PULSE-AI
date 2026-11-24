
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# DATOS PARA ENTRENAR EL MODELO
mensajes = [
    "Hola como estas",
    "Nos vemos mañana en clases",
    "Gracias por ayudarme",
    "Puedes enviarme el documento",
    "Estoy estudiando para el examen",
    "El profesor envio la tarea",
    "Nos encontramos en la universidad",
    "Feliz cumpleaños",
    "Ya llegue a casa",
    "Te llamo mas tarde",
    "Gana dinero gratis",
    "Has ganado un premio",
    "Reclama tu regalo ahora",
    "Oferta exclusiva compra ya",
    "Haz clic aqui para ganar",
    "Dinero gratis disponible",
    "Gana un celular gratis",
    "Ultima oportunidad de ganar",
    "Recibe tu bono gratuito",
    "Compra ahora y gana un premio"
]

categorias = [
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1
]

# CREAR Y ENTRENAR EL MODELO
modelo = Pipeline([
    ("vectorizador", CountVectorizer(
        strip_accents="unicode",
        ngram_range=(1, 2)
    )),
    ("clasificador", MultinomialNB())
])

modelo.fit(mensajes, categorias)

# PAGINA PRINCIPAL
@app.route("/", methods=["GET", "POST"])
def inicio():

    resultado = None
    confianza = None
    mensaje = ""

    if request.method == "POST":

        mensaje = request.form.get("mensaje", "").strip()

        if mensaje:
            prediccion = modelo.predict([mensaje])[0]
            probabilidades = modelo.predict_proba([mensaje])[0]

            confianza = round(
                float(probabilidades[prediccion]) * 100, 2
            )

            if prediccion == 1:
                resultado = "SPAM"
            else:
                resultado = "NO SPAM"

    return render_template(
        "index.html",
        resultado=resultado,
        confianza=confianza,
        mensaje=mensaje
    )

# INICIAR SERVIDOR WEB
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
