"""El loop del agente, independiente del dominio.

`correr` recibe las fichas (esquemas JSON) y las funciones de un dominio, y se queda con todo lo que
no es del modelo: qué puede elegir, cómo se ejecuta cada llamada, qué vuelve a la ventana y cuándo
se frena. El Bloque 1 lo usa con las herramientas del Mundial; el Bloque 2, con las de comidas.

Cada decisión que no sale del enunciado tiene su motivo en decision_log.md (sección «Bloque 1»).
"""
from __future__ import annotations

import ast
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturoVencido
from dataclasses import dataclass, field
from typing import Callable

import llm

# --- configuración con valores por defecto (las claves nuevas son opcionales en config.toml) ------

DEFECTOS = {
    "reintentos": {"max_intentos": 3, "espera_base_s": 1.0},
    "herramientas": {"timeout_s": 20.0},
    "loop": {"max_recordatorios": 2, "aviso_ultimo_paso": True, "elementos_lista_max": 25},
}


def _opcion(config: dict, seccion: str, clave: str):
    return config.get(seccion, {}).get(clave, DEFECTOS[seccion][clave])


# --- estado de una corrida ----------------------------------------------------------------------

@dataclass
class Estado:
    paso: int = 0
    tokens: int = 0
    recordatorios: int = 0
    ids_usados: set = field(default_factory=set)
    cache: dict = field(default_factory=dict)   # (nombre, argumentos canónicos) -> texto del resultado
    inicio: float = field(default_factory=time.monotonic)


class Fin(Exception):
    """Corta el loop con (estado, respuesta)."""

    def __init__(self, estado: str, respuesta: str | None = None, motivo: str | None = None):
        super().__init__(motivo or estado)
        self.estado, self.respuesta, self.motivo = estado, respuesta, motivo


# --- validación de argumentos contra la ficha ---------------------------------------------------

def _tipos(esquema: dict) -> list[str]:
    tipo = esquema.get("type")
    return tipo if isinstance(tipo, list) else [tipo] if tipo else []


def _coercionar(valor, esquema: dict):
    """Convierte lo que el modelo suele escribir mal y es inequívoco ("1978" → 1978). Si no, ValueError."""
    tipos = _tipos(esquema)
    if not tipos:
        return valor
    if valor is None:
        if "null" in tipos:
            return None
        raise ValueError("no puede ser null")
    if "integer" in tipos:
        if isinstance(valor, bool):
            raise ValueError("tiene que ser un entero, no un booleano")
        if isinstance(valor, int):
            return valor
        if isinstance(valor, float) and valor.is_integer():
            return int(valor)
        if isinstance(valor, str) and re.fullmatch(r"\s*-?\d+\s*", valor):
            return int(valor)
    if "number" in tipos:
        if isinstance(valor, (int, float)) and not isinstance(valor, bool):
            return valor
        if isinstance(valor, str):
            try:
                return float(valor.replace(",", "."))
            except ValueError:
                pass
    if "boolean" in tipos:
        if isinstance(valor, bool):
            return valor
        if isinstance(valor, str) and valor.strip().lower() in ("true", "false"):
            return valor.strip().lower() == "true"
    if "string" in tipos:
        if isinstance(valor, str):
            return valor
        if isinstance(valor, (int, float)) and not isinstance(valor, bool):
            return str(valor)
    if "array" in tipos and isinstance(valor, list):
        return valor
    if "object" in tipos and isinstance(valor, dict):
        return valor
    raise ValueError(f"tiene que ser {' o '.join(tipos)}, y llegó {type(valor).__name__} ({valor!r})")


def validar_argumentos(ficha: dict, argumentos) -> tuple[dict | None, str | None]:
    """Devuelve (argumentos limpios, None) o (None, mensaje de error para el modelo)."""
    if isinstance(argumentos, str):
        try:
            argumentos = json.loads(argumentos) if argumentos.strip() else {}
        except json.JSONDecodeError as error:
            return None, f"los argumentos no son JSON válido ({error.msg}). Mandalos como un objeto JSON."
    if not isinstance(argumentos, dict):
        return None, "los argumentos tienen que ser un objeto JSON con nombre → valor."
    parametros = ficha["function"]["parameters"]
    propiedades = parametros.get("properties", {})
    sobrantes = sorted(set(argumentos) - set(propiedades))
    if sobrantes:
        return None, f"parámetros que no existen: {sobrantes}. Los válidos son: {sorted(propiedades)}."
    faltantes = [p for p in parametros.get("required", []) if p not in argumentos]
    if faltantes:
        return None, f"faltan parámetros obligatorios: {faltantes}."
    limpios = {}
    for nombre, valor in argumentos.items():
        try:
            limpios[nombre] = _coercionar(valor, propiedades[nombre])
        except ValueError as error:
            return None, f"el parámetro «{nombre}» {error}."
    return limpios, None


# --- ejecución de herramientas ------------------------------------------------------------------

_EJECUTOR = ThreadPoolExecutor(max_workers=4)


def ejecutar(funcion: Callable, argumentos: dict, timeout_s: float) -> tuple[object, bool]:
    """Devuelve (resultado, ok). Nunca levanta: las fallas vuelven al modelo como {"error": ...}."""
    futuro = _EJECUTOR.submit(funcion, **argumentos)
    try:
        resultado = futuro.result(timeout=timeout_s)
    except FuturoVencido:
        return {"error": f"la herramienta tardó más de {timeout_s:g} s y se cortó. Probá con un filtro más chico."}, False
    except Exception as error:   # noqa: BLE001 — cualquier falla de la herramienta vuelve al modelo
        return {"error": f"la herramienta falló: {type(error).__name__}: {error}"}, False
    if isinstance(resultado, dict) and "error" in resultado:
        return resultado, False
    if resultado in (None, [], {}, ""):
        return {"resultado": [], "aviso": "La consulta no devolvió nada. Revisá los argumentos o "
                                          "concluí que el dato no está."}, True
    return resultado, True


def a_texto(resultado, tope_caracteres: int, elementos_max: int) -> tuple[str, bool]:
    """Serializa el resultado para la ventana. Recorta listas por elementos (dejando el total) antes
    de recortar por caracteres, para no devolver JSON partido a la mitad."""
    truncado = False
    if isinstance(resultado, list) and len(resultado) > elementos_max:
        resultado = {"total_elementos": len(resultado), "mostrados": elementos_max,
                     "aviso": f"Lista recortada: se muestran {elementos_max} de {len(resultado)}. "
                              "El total de elementos es exacto.",
                     "elementos": resultado[:elementos_max]}
        truncado = True
    texto = json.dumps(resultado, ensure_ascii=False, default=str)
    if len(texto) > tope_caracteres:
        texto = texto[:tope_caracteres] + f" …[recortado: {len(texto)} caracteres en total]"
        truncado = True
    return texto, truncado


# --- llamadas a terminar escritas como texto ---------------------------------------------------

TERMINAR_EN_TEXTO = re.compile(r"terminar\s*\((.*)\)", re.DOTALL)


def terminar_en_texto(contenido: str) -> dict | None:
    """Rescata `terminar("12", "motivo")` o `terminar(respuesta=..., motivo=...)` escrito como texto."""
    coincidencia = TERMINAR_EN_TEXTO.search(contenido or "")
    if not coincidencia:
        return None
    try:
        llamada = ast.parse(f"f({coincidencia.group(1)})", mode="eval").body
        posicionales = [ast.literal_eval(a) for a in llamada.args]
        nombrados = {k.arg: ast.literal_eval(k.value) for k in llamada.keywords}
    except (SyntaxError, ValueError):
        return None
    argumentos = dict(zip(("respuesta", "motivo"), posicionales)) | nombrados
    return argumentos if "respuesta" in argumentos else None


def llamadas_en_texto(contenido: str) -> list[dict]:
    """Rescata llamadas que el modelo escribió como texto en vez de como tool_calls.

    Reconoce objetos JSON con "name" y "arguments" (lo que qwen3 escribe cuando se le escapa el
    formato) y `terminar(...)` con sintaxis de función.
    """
    contenido = contenido or ""
    rescatadas, decodificador, i = [], json.JSONDecoder(), 0
    while (i := contenido.find("{", i)) != -1:
        try:
            objeto, fin = decodificador.raw_decode(contenido, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(objeto, dict) and isinstance(objeto.get("name"), str) and "arguments" in objeto:
            rescatadas.append({"function": {"name": objeto["name"], "arguments": objeto["arguments"]}})
        i = fin
    if not rescatadas and (terminar := terminar_en_texto(contenido)) is not None:
        rescatadas.append({"function": {"name": "terminar", "arguments": terminar}})
    return rescatadas


def normalizar_respuesta(valor) -> str | None:
    if valor is None:
        return None
    if isinstance(valor, list):
        valor = "; ".join(str(v) for v in valor)
    texto = str(valor).strip()
    return None if texto.lower() in ("", "null", "none", "n/a") else texto


# --- llamada al modelo con reintentos -----------------------------------------------------------

def llamar_modelo(chat: Callable, mensajes: list, fichas: list, config: dict, estado: Estado, registro,
                  dormir: Callable = time.sleep) -> llm.Respuesta:
    m = config["modelo"]
    max_intentos = _opcion(config, "reintentos", "max_intentos")
    espera = _opcion(config, "reintentos", "espera_base_s")
    for intento in range(1, max_intentos + 1):
        try:
            return chat(mensajes, modelo=m["nombre"], num_ctx=m["num_ctx"], temperatura=m["temperatura"],
                        semilla=m["semilla"], think=m["think"], num_predict=m["num_predict"],
                        herramientas=fichas, timeout_s=m["timeout_s"])
        except llm.ContextOverflow:
            # Reintentar con la misma ventana daría lo mismo: lo resuelve el loop compactando.
            raise
        except llm.LLMError as error:
            if error.estado is not None and 400 <= error.estado < 500:
                raise Fin("error", motivo=f"el servidor rechazó el pedido: {error}") from error
            registro.evento("reintento_modelo", paso=estado.paso, motivo=str(error)[:300], intento=intento)
            if intento == max_intentos:
                raise Fin("error", motivo=f"el servidor falló {max_intentos} veces: {error}") from error
            dormir(espera * 2 ** (intento - 1))
    raise AssertionError("inalcanzable")


# --- compactación ---------------------------------------------------------------------------------

OMITIDO = ("[resultado omitido para liberar contexto. Si todavía lo necesitás, volvé a pedirlo, "
           "idealmente con un filtro más chico]")


def compactar(mensajes: list[dict], minimo: int = 200) -> int:
    """Reemplaza los resultados de herramientas viejos (todos menos los del último turno) por un aviso.
    Si no hay viejos, recorta también los del último turno. Devuelve cuántos reemplazó."""
    ultimo_asistente = max((i for i, m in enumerate(mensajes) if m["role"] == "assistant"), default=-1)
    for hasta in (ultimo_asistente, len(mensajes)):
        reemplazados = 0
        for i, mensaje in enumerate(mensajes[:hasta]):
            if mensaje["role"] == "tool" and len(mensaje["content"]) > minimo:
                mensajes[i] = mensaje | {"content": OMITIDO}
                reemplazados += 1
        if reemplazados:
            return reemplazados
    return 0


# --- el loop ------------------------------------------------------------------------------------

RECORDATORIO = ("No llamaste a ninguna herramienta. Si ya tenés la respuesta, llamá a "
                "terminar(respuesta, motivo); si los datos no alcanzan, terminar(null, motivo). "
                "Si te falta un dato, llamá a la herramienta que corresponda.")
ULTIMO_PASO = ("Este es tu último paso: llamá ahora a terminar con lo que sabés, o con respuesta null "
               "si los datos que juntaste no alcanzan.")


def _id_unico(estado: Estado, propuesto: str | None) -> str:
    base = propuesto or f"call_{estado.paso}_{len(estado.ids_usados) + 1}"
    identificador, n = base, 1
    while identificador in estado.ids_usados:
        n += 1
        identificador = f"{base}_{n}"
    estado.ids_usados.add(identificador)
    return identificador


def correr(pregunta: str, config: dict, registro, *, sistema: str, fichas: list[dict],
           herramientas: dict[str, Callable], chat: Callable = llm.chat,
           dormir: Callable = time.sleep) -> tuple[str, str | None, int]:
    """Devuelve (estado, respuesta, pasos). Estados: respondida, negada, tope_pasos, tope_tokens, error."""
    estado = Estado()
    try:
        return _correr(pregunta, config, registro, sistema, fichas, herramientas, chat, dormir, estado)
    except Fin as fin:
        if fin.motivo:
            registro.evento("motivo_fin", paso=max(estado.paso, 1), motivo=fin.motivo)
        return fin.estado, fin.respuesta, estado.paso


def _correr(pregunta, config, registro, sistema, fichas, herramientas, chat, dormir, estado):
    frenos, m = config["frenos"], config["modelo"]
    tope_texto = config["contexto"]["tope_resultado_caracteres"]
    elementos_max = _opcion(config, "loop", "elementos_lista_max")
    timeout_herramienta = _opcion(config, "herramientas", "timeout_s")
    fichas_por_nombre = {f["function"]["name"]: f for f in fichas}
    solo_terminar = [fichas_por_nombre["terminar"]]
    mensajes = [{"role": "system", "content": sistema}, {"role": "user", "content": pregunta}]

    while True:
        if estado.paso >= frenos["max_pasos"]:
            raise Fin("tope_pasos", motivo=f"se llegó a {frenos['max_pasos']} llamadas al modelo sin terminar")
        if estado.tokens >= frenos["max_tokens"]:
            raise Fin("tope_tokens", motivo=f"se gastaron {estado.tokens} tokens (tope {frenos['max_tokens']})")
        if "max_seg" in frenos and time.monotonic() - estado.inicio >= frenos["max_seg"]:
            raise Fin("error", motivo=f"se pasó el tope de {frenos['max_seg']} s")
        estado.paso += 1
        ultimo = estado.paso == frenos["max_pasos"]
        if ultimo and _opcion(config, "loop", "aviso_ultimo_paso"):
            mensajes.append({"role": "user", "content": ULTIMO_PASO})
        disponibles = solo_terminar if ultimo else fichas

        try:
            respuesta = llamar_modelo(chat, mensajes, disponibles, config, estado, registro, dormir)
        except llm.ContextOverflow as error:
            reemplazados = compactar(mensajes)
            if not reemplazados:
                raise Fin("error", motivo=f"el contexto no entra en num_ctx={m['num_ctx']}: {error}") from error
            estado.cache.clear()   # lo omitido ya no está «más arriba»: se puede volver a pedir
            registro.evento("compactacion", paso=estado.paso, reemplazados=reemplazados, motivo=str(error)[:200])
            try:
                respuesta = llamar_modelo(chat, mensajes, disponibles, config, estado, registro, dormir)
            except llm.ContextOverflow as error2:
                raise Fin("error", motivo=f"no entra en num_ctx aun compactando: {error2}") from error2
        estado.tokens += respuesta.tokens_entrada + respuesta.tokens_salida
        registro.evento("llamada_modelo", paso=estado.paso, modelo=m["nombre"], num_ctx=m["num_ctx"],
                        tokens_entrada=respuesta.tokens_entrada, tokens_salida=respuesta.tokens_salida,
                        duracion_ms=respuesta.duracion_ms, razon_fin=respuesta.razon_fin,
                        tokens_acumulados=estado.tokens)
        mensaje = respuesta.mensaje
        llamadas = mensaje.get("tool_calls") or []
        mensajes.append({k: v for k, v in mensaje.items() if k in ("role", "content", "tool_calls")})

        if not llamadas and (llamadas := llamadas_en_texto(mensaje.get("content", ""))):
            registro.evento("llamadas_rescatadas", paso=estado.paso, cantidad=len(llamadas),
                            nombres=[_nombre(c) for c in llamadas])
            mensajes[-1] = {"role": "assistant", "content": "", "tool_calls": llamadas}
        if not llamadas:
            if estado.recordatorios >= _opcion(config, "loop", "max_recordatorios"):
                raise Fin("error", motivo="el modelo respondió en texto sin llamar a terminar, "
                                          f"{estado.recordatorios + 1} veces")
            estado.recordatorios += 1
            registro.evento("recordatorio", paso=estado.paso, contenido=(mensaje.get("content") or "")[:300])
            mensajes.append({"role": "user", "content": RECORDATORIO})
            continue

        hay_otras = any(_nombre(c) != "terminar" for c in llamadas)
        final = None
        for llamada in llamadas:
            nombre, argumentos_crudos = _nombre(llamada), llamada.get("function", {}).get("arguments", {})
            identificador = _id_unico(estado, llamada.get("id"))
            registro.evento("llamada_herramienta", paso=estado.paso, id=identificador, nombre=nombre or "?",
                            argumentos=argumentos_crudos if isinstance(argumentos_crudos, (dict, str))
                            else str(argumentos_crudos))
            inicio = time.perf_counter()
            texto, ok, truncado, argumentos = _atender(nombre, argumentos_crudos, fichas_por_nombre,
                                                        disponibles, herramientas, hay_otras, estado,
                                                        tope_texto, elementos_max, timeout_herramienta)
            registro.evento("resultado_herramienta", paso=estado.paso, id=identificador, ok=ok,
                            truncado=truncado, caracteres=len(texto),
                            duracion_ms=round((time.perf_counter() - inicio) * 1000))
            if nombre == "terminar" and ok:
                final = argumentos
            else:
                mensajes.append({"role": "tool", "tool_name": nombre or "?", "tool_call_id": identificador,
                                 "content": texto})
        if final is not None:
            _terminar(final)


def _nombre(llamada: dict) -> str | None:
    return llamada.get("function", {}).get("name")


def _terminar(argumentos: dict):
    respuesta = normalizar_respuesta(argumentos.get("respuesta"))
    raise Fin("respondida" if respuesta is not None else "negada", respuesta,
              motivo=str(argumentos.get("motivo") or "") or None)


def _atender(nombre, argumentos_crudos, fichas_por_nombre, disponibles, herramientas, hay_otras, estado,
             tope_texto, elementos_max, timeout_s):
    """Devuelve (texto para el modelo, ok, truncado, argumentos limpios)."""
    def error(mensaje):
        return json.dumps({"error": mensaje}, ensure_ascii=False), False, False, None

    validas = [f["function"]["name"] for f in disponibles]
    if nombre not in validas:
        return error(f"la herramienta «{nombre}» no existe o no está disponible ahora. Usá una de: {validas}.")
    argumentos, problema = validar_argumentos(fichas_por_nombre[nombre], argumentos_crudos)
    if problema:
        return error(f"{nombre}: {problema}")
    if nombre == "terminar":
        if hay_otras:
            return error("llamaste a terminar junto con otras herramientas. Mirá primero sus resultados "
                         "y después llamá a terminar sola.")
        return "", True, False, argumentos
    clave = (nombre, json.dumps(argumentos, sort_keys=True, ensure_ascii=False))
    if clave in estado.cache:
        aviso = ("Ya hiciste exactamente esta llamada y su resultado está más arriba en la conversación. "
                 "No la repitas: usá otros argumentos, calcular con los datos que ya tenés, o terminá "
                 "(con null si los datos no alcanzan).")
        return json.dumps({"aviso": aviso}, ensure_ascii=False), True, False, argumentos
    resultado, ok = ejecutar(herramientas[nombre], argumentos, timeout_s)
    texto, truncado = a_texto(resultado, tope_texto, elementos_max)
    if ok:
        estado.cache[clave] = True
    return texto, ok, truncado, argumentos
