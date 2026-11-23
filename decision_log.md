# Registro de decisiones

Cada entrada lleva fecha, qué se decidió y por qué. Las ediciones del conjunto de evaluación congelado
también van acá.

## 2026-11-23 · Presupuesto y plan (primera entrada)

> **A completar por el grupo:** horas por integrante hasta el 29/11 y reparto de tareas.

- **Cómputo:** desarrollo local en una RTX 3050 de 6 GB (i5-9400F, 16 GB RAM, Windows 11) con
  `qwen3:8b`. Los tiempos que se reporten para comparar se miden en Colab T4, como pide el enunciado.
- **Costo estimado del Bloque 1:** en la primera prueba local, unas 2,5 llamadas al modelo y ~4.000
  tokens por tarea fácil, entre 10 y 15 s cada una. Las 30 tareas son ~15 min por corrida en local.
- **Plan:** Bloque 1 primero, porque el loop se reutiliza en el Bloque 2 (`harness/loop.py` no
  depende del dominio). Después, herramientas y conjunto de evaluación de comidas, líneas de base y
  las dos decisiones medidas.
- **Versión de Ollama:** local es 0.40.17; el kit fija 0.40.10. Se declara en el informe. Las
  corridas en Colab usan la del cuaderno de arranque.
- **Repositorio:** el repo privado se creó vacío (no con *Use this template*); el contenido del kit
  se trajo con un remoto `kit` y se conserva su historial.

## 2026-11-23 · Bloque 1: decisiones de diseño del loop

El loop vive en `harness/loop.py` y `agente.py` solo le pasa el prompt de sistema y las
herramientas del Mundial. Decisiones que no salen del enunciado:

1. **Validar los argumentos contra la ficha antes de ejecutar**, con coerción solo cuando es
   inequívoca (`"1978"` → `1978`, `"true"` → `true`). *Motivo:* el modelo a veces manda enteros
   como texto; rechazarlos gasta un paso entero sin ganar nada. Lo ambiguo vuelve como error con los
   parámetros válidos.
2. **Recortar listas por elementos antes que por caracteres, dejando `total_elementos`.** *Motivo:*
   `partidos(1986)` devuelve 52 partidos; un corte por caracteres deja JSON partido y el modelo
   cuenta mal los que ve. Con el total exacto la pregunta «¿cuántos partidos?» se responde sin
   contar.
3. **En el último paso solo se ofrece `terminar`**, con un aviso. *Motivo:* sin esto, una corrida que
   llega al tope termina sin respuesta aunque el modelo ya tuviera los datos.
4. **Llamada repetida idéntica → se devuelve el resultado anterior con un aviso**, sin volver a
   ejecutar. *Motivo:* corta bucles donde el modelo repite la misma consulta esperando otra cosa.
5. **`terminar` junto con otras herramientas en el mismo turno no termina**: las otras se ejecutan
   y `terminar` vuelve como error. *Motivo:* si el modelo decide la respuesta antes de ver los
   resultados que pidió, la respuesta no sale de los datos.
6. **Texto sin llamadas:** si el texto contiene `terminar(...)`, se rescata; si no, se le recuerda
   que llame a `terminar` hasta 2 veces y después se corta con `error`. *Motivo:* el README del kit
   avisa que los modelos chicos escriben `terminar` como texto; contestar en prosa no es una
   respuesta verificable.
7. **Reintentos solo para errores del servidor (5xx o conexión)**, con espera exponencial y 3
   intentos. Los 4xx y `ContextOverflow` cortan sin reintentar. *Motivo:* con temperatura 0 y el
   mismo prompt, reintentar un pedido rechazado da el mismo rechazo.
8. **Las herramientas corren en un hilo con timeout (20 s)**; una excepción o un timeout vuelven al
   modelo como `{"error": ...}` y un resultado vacío vuelve con un aviso explícito. *Motivo:* el
   agente no se cae y el modelo puede decidir negarse con evidencia.
9. **Ids de llamada únicos en toda la traza**: se usan los de Ollama y, si faltan o se repiten, se
   generan. *Motivo:* el validador exige ids únicos y emparejados.
10. **Eventos extra en la traza** (`recordatorio`, `motivo_fin`, `razon_fin` y `tokens_acumulados`
    en `llamada_modelo`). *Motivo:* para leer después por qué terminó cada corrida.

Configuración nueva en `config.toml`: `[reintentos]`, `[herramientas]` y `[loop]`.

## 2026-11-23 · Bloque 1: primera prueba en vivo

4 tareas fáciles (mun-001, 003, 004, 006): 4 aciertos, 10 llamadas al modelo, 15.614 tokens. La
primera tarea tardó 110 s porque incluye la carga del modelo en la GPU.

## 2026-11-23 · Bloque 1: corrida v1 (30 tareas, local) y qué falló según las trazas

`trazas/b1_v1_local`: **23 de 30 aciertos** (fáciles 11/13, medias 7/10, difíciles 5/7), pero
**0 de 4 imposibles bien negadas**. 134 llamadas al modelo, 272.157 tokens, 829 s.

| Tarea | Resultado | Qué muestra la traza primero |
|---|---|---|
| mun-012 (campeón 2026, imposible) | error | El modelo escribe `{"name": "terminar", "arguments": {...}}` como **texto**. El rescate solo reconocía `terminar(...)`; tras 2 recordatorios, corte por texto sin terminar. |
| mun-023 (posesión 1978, imposible) | error | Dice en texto que no se puede, después llama 5 veces a `partidos(1978)`; cada repetición vuelve a meter ~4.000 caracteres y desborda `num_ctx=8192`. |
| mun-018 (penales 2022) | error | `partidos(2022)` tiene 64 partidos y se recorta a 25: los de penales (octavos en adelante) **quedan afuera**. Repite la llamada hasta desbordar el contexto. |
| mun-027 (goles de Francia 1998) | error | Mismo recorte; prueba `goles("France")` y `calcular("sumar goles de France...")`; desborda en el paso 13. |
| mun-013 (campeona femenina 2023, imposible) | invento | Ve que 2023 no está y responde el campeón de 2019. El motivo lo admite. |
| mun-022 (espectadores 1990, imposible) | invento | Responde 916000 sin que el número aparezca en ningún resultado. |
| mun-030 (goles de carrera de los 3 goleadores 2002) | error_respuesta | Pide los goles **del torneo 2002** en vez de los de carrera (18 en vez de 39). |

### Cambios para la v2

1. **Rescatar llamadas escritas como JSON en texto** (cualquier herramienta, no solo `terminar`), y
   seguir el loop como si fueran `tool_calls`. Queda un evento `llamadas_rescatadas`. → mun-012.
2. **Llamada repetida: aviso corto, sin reenviar el resultado.** Antes se reenviaba entero y era lo
   que llenaba la ventana. → mun-018, 023, 027.
3. **Compactar ante `ContextOverflow`** en vez de cortar: los resultados de herramientas viejos se
   reemplazan por un aviso (si no hay viejos, también los del último turno) y se reintenta una vez.
   Se limpia la caché de repetidas para que lo omitido se pueda volver a pedir.
4. **Herramienta `partidos` con filtros `equipo` y `solo_penales`.** Cambio de herramienta
   justificado: el recorte por elementos (decisión 2 de arriba) es correcto para contar, pero deja
   afuera los partidos que importan; con el filtro, el resultado entra entero. → mun-018, 027.

No se toca en el Bloque 1: los inventos de mun-013/022 y el error de lectura de mun-030. Son el
material para una de las decisiones medidas del Bloque 2 (verificar la respuesta contra los
resultados antes de terminar).

## 2026-11-23 · Bloque 1: corrida v2 (30 tareas, local)

`trazas/b1_v2_local`: **25 de 30** (fáciles 11/13, medias 8/10, difíciles 6/7), **1 de 4 imposibles
bien negadas**. 133 llamadas, 236.633 tokens (−13 %), 665 s (−20 %).

- **Arreglados:** mun-018 (5, con `partidos(2022, solo_penales=true)`), mun-027 (15, con
  `equipo="France"`) y mun-023, que ahora se niega: después de varias consultas a `partidos` llama
  `terminar("null", "No hay datos sobre la posesión de la pelota...")`; la traza muestra que ningún
  resultado tiene un campo de posesión. Es la tarea negada con traza que pide el enunciado.
- **mun-012:** el rescate de JSON en texto funciona (ya no termina en error), pero el modelo
  responde «Argentina» con un motivo que dice que 2026 no está en la base: pasa de `error` a
  `invento`. El arreglo de formato destapó un problema de fidelidad, que no es del loop.
- **Regresión en mun-016 (1 → 2):** las 6 llamadas son idénticas en v1 y v2; solo cambia la
  lectura final (dice que Messi no hizo goles en 2022). El único cambio que ve el modelo en esta
  tarea es la descripción nueva de `partidos` en las fichas. Con temperatura 0, un cambio en el
  prompt que no tiene que ver con la tarea alcanza para dar vuelta una respuesta. **Consecuencia
  para la evaluación:** una diferencia de 1 o 2 tareas entre dos versiones no prueba nada sin una
  cota de ruido (por ejemplo, varias semillas o perturbaciones neutras del prompt).
- **Siguen:** mun-013, 022 (inventos) y 030 (lee mal la pregunta).
