"""El loop con un modelo simulado: cada test guiona lo que «responde» el modelo y mira la traza."""
import json
import time

import pytest

import llm
from harness import loop
from herramientas.mundial import FICHAS
from tests.ayudas import leer
from trazas import esquema
from trazas.registro import Registro

CONFIG = {
    "modelo": {"nombre": "qwen3:8b", "num_ctx": 8192, "temperatura": 0, "semilla": 1, "think": False,
               "num_predict": 800, "timeout_s": 300},
    "frenos": {"max_pasos": 5, "max_tokens": 10_000},
    "contexto": {"tope_resultado_caracteres": 400},
    "reintentos": {"max_intentos": 3, "espera_base_s": 0},
    "herramientas": {"timeout_s": 0.5},
    "loop": {"max_recordatorios": 1, "aviso_ultimo_paso": True, "elementos_lista_max": 5},
}


def herramientas_falsas():
    return {
        "calcular": lambda expresion: {"resultado": eval(expresion)},   # noqa: S307 — solo en tests
        "partidos": lambda torneo: [{"n": i, "estadio": "Estadio Azteca, Ciudad de México"} for i in range(52)],
        "goleadores": lambda torneo, n=5: (_ for _ in ()).throw(RuntimeError("se rompió la base")),
        "buscar_jugador": lambda nombre: time.sleep(2) or {"nombre": nombre},
        "goles": lambda jugador, torneo=None: [],
        "buscar_torneos": lambda campeon=None, anio=None, femenino=False: [{"torneo": 1978}],
    }


def llamada(nombre, argumentos, id=None):
    return {"function": {"name": nombre, "arguments": argumentos}, **({"id": id} if id else {})}


def modelo_guionado(*turnos, errores=()):
    """Cada turno es una lista de tool_calls o un texto. `errores` se levantan antes del primer turno."""
    cola_errores, cola = list(errores), list(turnos)
    vistos = []

    def chat(mensajes, **opciones):
        vistos.append({"mensajes": [dict(m) for m in mensajes], "herramientas": opciones["herramientas"]})
        if cola_errores:
            raise cola_errores.pop(0)
        turno = cola.pop(0)
        mensaje = {"role": "assistant", "content": turno} if isinstance(turno, str) else \
            {"role": "assistant", "content": "", "tool_calls": turno}
        return llm.Respuesta(mensaje, 100, 10, 5, "stop")

    chat.vistos = vistos
    return chat


def correr(tmp_path, chat, config=CONFIG):
    registro = Registro(tmp_path, "t")
    registro.evento("corrida_inicio", modelo="qwen3:8b", config=config)
    estado, respuesta, pasos = loop.correr("¿?", config, registro, sistema="s", fichas=FICHAS,
                                           herramientas=herramientas_falsas(), chat=chat, dormir=lambda s: None)
    registro.evento("corrida_fin", estado=estado, respuesta=respuesta, pasos=pasos)
    assert esquema.validar_archivo(registro.ruta) == []
    return estado, respuesta, pasos, leer(registro.ruta)


def resultados_al_modelo(chat, turno):
    return [json.loads(m["content"]) for m in chat.vistos[turno]["mensajes"] if m["role"] == "tool"]


def test_ciclo_basico_herramienta_y_terminar(tmp_path):
    chat = modelo_guionado([llamada("calcular", {"expresion": "6+6"})],
                           [llamada("terminar", {"respuesta": "12", "motivo": "cuenta"})])
    estado, respuesta, pasos, eventos = correr(tmp_path, chat)
    assert (estado, respuesta, pasos) == ("respondida", "12", 2)
    assert resultados_al_modelo(chat, 1) == [{"resultado": 12}]


def test_respuesta_null_es_negada(tmp_path):
    chat = modelo_guionado([llamada("terminar", {"respuesta": None, "motivo": "no hay dato"})])
    assert correr(tmp_path, chat)[:2] == ("negada", None)


def test_herramienta_inexistente_y_argumentos_malos_vuelven_como_error(tmp_path):
    chat = modelo_guionado([llamada("volar", {}), llamada("calcular", {"x": 1}), llamada("goleadores", {"torneo": "mil"})],
                           [llamada("terminar", {"respuesta": "1", "motivo": "m"})])
    estado, *_ , eventos = correr(tmp_path, chat)
    errores = [r["error"] for r in resultados_al_modelo(chat, 1)]
    assert "no existe" in errores[0] and "no existen" in errores[1] and "torneo" in errores[2]
    assert [e["ok"] for e in eventos if e["evento"] == "resultado_herramienta"][:3] == [False] * 3
    assert estado == "respondida"


def test_argumentos_como_texto_json_y_coercion(tmp_path):
    chat = modelo_guionado([llamada("partidos", '{"torneo": "1986"}')],
                           [llamada("terminar", {"respuesta": 52, "motivo": "m"})])
    config = CONFIG | {"contexto": {"tope_resultado_caracteres": 5000}}
    estado, respuesta, *_ = correr(tmp_path, chat, config)
    resultado = resultados_al_modelo(chat, 1)[0]
    assert resultado["total_elementos"] == 52 and len(resultado["elementos"]) == 5
    assert respuesta == "52"


def test_json_roto_no_cae(tmp_path):
    chat = modelo_guionado([llamada("calcular", '{"expresion": "1+')],
                           [llamada("terminar", {"respuesta": "x", "motivo": "m"})])
    correr(tmp_path, chat)
    assert "JSON" in resultados_al_modelo(chat, 1)[0]["error"]


def test_excepcion_timeout_y_vacio(tmp_path):
    chat = modelo_guionado([llamada("goleadores", {"torneo": 1978}), llamada("buscar_jugador", {"nombre": "X"}),
                            llamada("goles", {"jugador": "X"})],
                           [llamada("terminar", {"respuesta": None, "motivo": "m"})])
    correr(tmp_path, chat)
    r = resultados_al_modelo(chat, 1)
    assert "RuntimeError" in r[0]["error"] and "tardó" in r[1]["error"] and r[2]["resultado"] == []


def test_texto_sin_herramienta_recuerda_y_despues_corta(tmp_path):
    chat = modelo_guionado("La respuesta es 12.", "Sigo pensando.")
    estado, *_ , eventos = correr(tmp_path, chat)
    assert estado == "error" and sum(e["evento"] == "recordatorio" for e in eventos) == 1


def test_terminar_escrito_como_texto_se_rescata(tmp_path):
    chat = modelo_guionado('terminar("12", "de la cuenta")')
    assert correr(tmp_path, chat)[:2] == ("respondida", "12")


def test_terminar_junto_a_otras_no_corta(tmp_path):
    chat = modelo_guionado([llamada("calcular", {"expresion": "1+1"}), llamada("terminar", {"respuesta": "9", "motivo": "m"})],
                           [llamada("terminar", {"respuesta": "2", "motivo": "m"})])
    assert correr(tmp_path, chat)[:2] == ("respondida", "2")


def test_llamada_repetida_avisa(tmp_path):
    chat = modelo_guionado([llamada("calcular", {"expresion": "1+1"})], [llamada("calcular", {"expresion": "1+1"})],
                           [llamada("terminar", {"respuesta": "2", "motivo": "m"})])
    correr(tmp_path, chat)
    assert "aviso" in resultados_al_modelo(chat, 2)[-1]


def test_tope_de_pasos_y_ultimo_paso_solo_terminar(tmp_path):
    turnos = [[llamada("calcular", {"expresion": f"{i}+1"})] for i in range(5)]
    chat = modelo_guionado(*turnos)
    estado, _, pasos, _ = correr(tmp_path, chat)
    assert (estado, pasos) == ("tope_pasos", 5)
    assert [f["function"]["name"] for f in chat.vistos[-1]["herramientas"]] == ["terminar"]


def test_tope_de_tokens(tmp_path):
    config = CONFIG | {"frenos": {"max_pasos": 50, "max_tokens": 250}}
    chat = modelo_guionado(*[[llamada("calcular", {"expresion": f"{i}+1"})] for i in range(10)])
    assert correr(tmp_path, chat, config)[0] == "tope_tokens"


def test_reintenta_errores_del_servidor(tmp_path):
    chat = modelo_guionado([llamada("terminar", {"respuesta": "1", "motivo": "m"})],
                           errores=[llm.LLMError("caído", 500), llm.LLMError("sin respuesta")])
    estado, *_ , eventos = correr(tmp_path, chat)
    assert estado == "respondida" and [e["intento"] for e in eventos if e["evento"] == "reintento_modelo"] == [1, 2]


def test_se_rinde_despues_de_los_reintentos(tmp_path):
    chat = modelo_guionado(errores=[llm.LLMError("caído", 503)] * 3)
    assert correr(tmp_path, chat)[0] == "error"


def test_contexto_excedido_no_reintenta(tmp_path):
    chat = modelo_guionado(errores=[llm.ContextOverflow("no entra", 400)])
    estado, *_ , eventos = correr(tmp_path, chat)
    assert estado == "error" and not any(e["evento"] == "reintento_modelo" for e in eventos)


def test_ids_repetidos_del_servidor_se_desambiguan(tmp_path):
    chat = modelo_guionado([llamada("calcular", {"expresion": "1+1"}, id="a")],
                           [llamada("calcular", {"expresion": "2+1"}, id="a")],
                           [llamada("terminar", {"respuesta": "3", "motivo": "m"}, id="a")])
    eventos = correr(tmp_path, chat)[3]
    ids = [e["id"] for e in eventos if e["evento"] == "llamada_herramienta"]
    assert len(ids) == len(set(ids)) == 3


@pytest.mark.parametrize("valor,esperado", [("null", None), ("", None), (12, "12"), (["a", "b"], "a; b")])
def test_normalizar_respuesta(valor, esperado):
    assert loop.normalizar_respuesta(valor) == esperado


def test_llamada_escrita_como_json_en_texto_se_rescata(tmp_path):
    chat = modelo_guionado('{"name": "calcular", "arguments": {"expresion": "2+2"}}',
                           '<tool_call>{"name": "terminar", "arguments": {"respuesta": "4", "motivo": "m"}}</tool_call>')
    estado, respuesta, _, eventos = correr(tmp_path, chat)
    assert (estado, respuesta) == ("respondida", "4")
    assert resultados_al_modelo(chat, 1) == [{"resultado": 4}]
    assert sum(e["evento"] == "llamadas_rescatadas" for e in eventos) == 2


def test_contexto_excedido_compacta_y_sigue(tmp_path):
    chat = modelo_guionado([llamada("partidos", {"torneo": 1986})], [llamada("calcular", {"expresion": "1+1"})],
                           [llamada("terminar", {"respuesta": "2", "motivo": "m"})])
    original = chat

    def chat_que_desborda(mensajes, **opciones):
        if len(original.vistos) == 2 and not any(m["content"] == loop.OMITIDO for m in mensajes if m["role"] == "tool"):
            original.vistos.append({"mensajes": mensajes, "herramientas": opciones["herramientas"]})
            raise llm.ContextOverflow("no entra", 400)
        return original(mensajes, **opciones)

    chat_que_desborda.vistos = original.vistos
    estado, respuesta, _, eventos = correr(tmp_path, chat_que_desborda)
    assert (estado, respuesta) == ("respondida", "2")
    assert any(e["evento"] == "compactacion" for e in eventos)
