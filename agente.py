"""Tu loop. El kit trae todo lo demás; acá falta `bucle`.

    uv run agente.py -p "¿Qué edad tenía el goleador del Mundial 1978 el día de la final?" --log trazas/mi_corrida
    uv run agente.py -p "..." --tarea mun-024 --log trazas/mi_corrida --config config.toml

Ya está hecho: leer y validar `config.toml`, abrir la traza y escribir `corrida_inicio` con la
configuración y la versión de Ollama, y escribir `corrida_fin` cuando `bucle` devuelve.
Sin implementar `bucle`, esto termina con estado `error` y no responde nada.

Lo que tiene que resolver `bucle` (Bloque 1, sección 4 del enunciado):
  1. el ciclo: llamar al modelo, ejecutar cada llamada a herramienta y devolver un resultado por
     llamada, hasta que el modelo termine;
  2. el contrato de herramientas: `FICHAS` trae los esquemas y `HERRAMIENTAS` las funciones;
  3. la salida mal formada: JSON roto, herramienta inexistente, argumentos del tipo equivocado;
  4. las herramientas que fallan: excepciones, resultados vacíos, timeouts;
  5. el contexto: recortar los resultados largos (`contexto.tope_resultado_caracteres`);
  6. los reintentos ante errores del servidor del modelo, con límite;
  7. la parada: tope de pasos y de tokens, y `terminar(respuesta, motivo)`, con `respuesta = None`
     cuando los datos no alcanzan;
  8. las trazas: un evento por cada cosa que pasa, con el esquema de la sección 7 del enunciado.

Qué usar:
  llm.chat(mensajes, modelo=..., num_ctx=..., herramientas=FICHAS)  -> llm.Respuesta
      una llamada; levanta llm.LLMError (y llm.ContextOverflow si el prompt no entra)
  registro.evento("llamada_modelo", paso=1, modelo=..., num_ctx=..., tokens_entrada=...,
                  tokens_salida=..., duracion_ms=...)
  HERRAMIENTAS[nombre](**argumentos)  -> datos, o {"error": ...}
"""
import argparse
import sys
from pathlib import Path

import config as configuracion
import llm
from harness import loop
from herramientas.mundial import FICHAS, HERRAMIENTAS
from trazas.registro import Registro, SinRegistro

RAIZ = Path(__file__).resolve().parent


SISTEMA = """Sos un agente que responde preguntas sobre los Mundiales de fútbol usando SOLO los datos \
que devuelven las herramientas (la Fjelstul World Cup Database). No uses lo que sabés de memoria: \
todo dato de la respuesta tiene que haber salido de una herramienta.

Reglas:
- Los nombres de selecciones y países están en inglés (Argentina, West Germany, United States).
- Si una búsqueda de jugador es ambigua, elegí el candidato por su id (P-xxxxx) y volvé a llamar con el id.
- Para cualquier cuenta (sumas, restas, edades, diferencias de fechas) usá calcular; no hagas cuentas de cabeza.
- Para contar elementos de una lista larga, usá el campo total_elementos si aparece.
- Cuando tengas la respuesta, llamá a terminar(respuesta, motivo) con la respuesta en el formato que pide la pregunta.
- Si los datos de las herramientas no alcanzan para responder (el dato no existe en la base, o el torneo \
o el jugador no están), llamá a terminar con respuesta null y un motivo que diga qué dato falta. \
Nunca inventes una respuesta."""


def bucle(pregunta: str, config: dict, registro) -> tuple[str, str | None, int]:
    """Devuelve (estado, respuesta, pasos). Estados: respondida, negada, tope_pasos, tope_tokens, error."""
    return loop.correr(pregunta, config, registro, sistema=SISTEMA, fichas=FICHAS, herramientas=HERRAMIENTAS)


def entorno_del_servidor(modelo: str) -> dict:
    """La versión de Ollama y el digest del modelo; null si el servidor no responde."""
    try:
        return {"ollama_version": llm.version(), "modelo_digest": llm.digest(modelo)}
    except llm.LLMError:
        return {"ollama_version": None, "modelo_digest": None}


def main() -> int:
    parser = argparse.ArgumentParser(description="Responde una pregunta sobre los Mundiales con herramientas.")
    parser.add_argument("-p", "--pregunta", required=True)
    parser.add_argument("--tarea", help="id de la tarea, para la traza")
    parser.add_argument("--log", type=Path, help="carpeta de trazas; sin esto no se escribe ninguna")
    parser.add_argument("--config", type=Path, default=configuracion.CONFIG)
    argumentos = parser.parse_args()

    config = configuracion.cargar(argumentos.config)
    registro = Registro(argumentos.log, argumentos.tarea) if argumentos.log else SinRegistro()
    modelo = config["modelo"]["nombre"]
    registro.evento("corrida_inicio", modelo=modelo, config=config | entorno_del_servidor(modelo))
    try:
        estado, respuesta, pasos = bucle(argumentos.pregunta, config, registro)
    except Exception as error:   # noqa: BLE001 — un bug del harness igual cierra la traza
        registro.evento("corrida_fin", estado="error", respuesta=None, pasos=0,
                        motivo=f"{type(error).__name__}: {error}")
        print(error, file=sys.stderr)
        return 1
    registro.evento("corrida_fin", estado=estado, respuesta=respuesta, pasos=pasos)
    print(respuesta if respuesta is not None else f"({estado})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
