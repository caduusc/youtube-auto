"""Mostra o que cada limiar de corte removeria do SEU jeito de falar.

    uv run python tools/calibrar_corte.py work/<slug>/transcript.json

Nao altera nada. Serve para escolher `trim.pause_min_seconds` e
`trim.filler_*` com dados em vez de palpite.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.config import TrimConfig  # noqa: E402
from pipeline.schemas import Transcript  # noqa: E402
from pipeline.trim import build_plan, filler_floor, normalize, token_medians  # noqa: E402

FPS = 30.0


def main(path: Path) -> None:
    transcript = Transcript.model_validate(json.loads(path.read_text(encoding="utf-8")))
    words = [w for s in transcript.segments for w in s.words]
    words.sort(key=lambda w: w.start)
    if not words:
        print("O transcript nao tem timestamp por palavra.")
        return

    print(f"\n{path}")
    print(f"{transcript.duration:.1f}s, {len(transcript.segments)} segmentos, "
          f"{len(words)} palavras\n")

    # --- pausas: quanto cada limiar removeria -----------------------------
    gaps = sorted(
        (b.start - a.end for a, b in zip(words, words[1:]) if b.start > a.end),
        reverse=True,
    )
    print("  Pausas: o que cada limiar de `pause_min_seconds` removeria")
    print("  " + "-" * 62)
    print(f"  {'limiar':>8}  {'pausas':>7}  {'removido':>9}  {'do video':>9}")
    for limiar in (0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5):
        alvo = [g for g in gaps if g >= limiar]
        # o corte deixa a margem nas duas pontas
        removido = sum(max(0.0, g - 2 * 0.12) for g in alvo)
        print(f"  {limiar:>7.1f}s  {len(alvo):>7}  {removido:>8.1f}s  "
              f"{removido / transcript.duration:>8.1%}")

    if gaps:
        print(f"\n  maior pausa: {gaps[0]:.2f}s | "
              f"mediana dos intervalos: {sorted(gaps)[len(gaps) // 2]:.2f}s | "
              f"{len(gaps)} intervalos > 0")

    # --- aperto: o outro mecanismo ----------------------------------------
    # Se a maior pausa da fala nao chega perto do limiar de remocao, remover
    # nao tem materia-prima e a tabela acima ja disse isso. O aperto nao
    # depende de haver trecho morto: ele reduz TODO intervalo ao teto.
    print("\n  Aperto: o que cada `pause_max_seconds` removeria")
    print("  " + "-" * 62)
    print(f"  {'teto':>8}  {'emendas':>7}  {'removido':>9}  {'do video':>9}")
    for teto in (0.40, 0.35, 0.30, 0.25, 0.20, 0.15):
        alvo = [g for g in gaps if g > teto]
        removido = sum(g - teto for g in alvo)
        print(f"  {teto:>7.2f}s  {len(alvo):>7}  {removido:>8.1f}s  "
              f"{removido / transcript.duration:>8.1%}")
    print("  Cada emenda e um ponto de corte no filtergraph e um risco de")
    print("  artefato — e a coluna que decide se vale, nao so a porcentagem.")

    # --- hesitacoes: existem no transcript? -------------------------------
    rules = TrimConfig()
    wanted = {normalize(f) for f in rules.fillers}
    candidatos = []
    for i, w in enumerate(words):
        if normalize(w.word) not in wanted:
            continue
        antes = w.start - (words[i - 1].end if i else 0.0)
        depois = (words[i + 1].start if i + 1 < len(words) else transcript.duration) - w.end
        candidatos.append((w.word.strip(), w.end - w.start, antes, depois, w.start))

    print(f"\n  Hesitacoes: tokens da lista encontrados no transcript")
    print("  " + "-" * 62)
    if not candidatos:
        print("  Nenhum token da lista aparece. O Whisper tende a limpar")
        print("  disfluencia na transcricao, entao o som pode existir no audio")
        print("  e nao aparecer no texto. Nesse caso o corte de hesitacao nao")
        print("  tem o que cortar, e o unico lever que sobra e `pause_min_seconds`.")
    else:
        medians = token_medians(words)

        def veredito(token: str, dur: float, antes: float) -> tuple[bool, str]:
            """O mesmo par de testes de `find_cuts`, e qual deles reprovou."""
            if antes < rules.filler_silence_seconds:
                return False, "sem silencio antes"
            piso = filler_floor(normalize(token), medians, rules)
            if dur < piso:
                return False, f"curto p/ o piso {piso:.2f}s"
            return True, ""

        print(f"  {'token':>10}  {'dur':>5}  {'sil.antes':>9}  {'piso':>6}  veredito")
        for token, dur, antes, _depois, quando in sorted(candidatos, key=lambda c: -c[1])[:20]:
            corta, porque = veredito(token, dur, antes)
            piso = filler_floor(normalize(token), medians, rules)
            print(f"  {token:>10}  {dur:>4.2f}s  {antes:>8.2f}s  {piso:>5.2f}s  "
                  f"{'CORTA' if corta else 'mantem':<6}  {porque:<22} ({quando:.1f}s)")

        cortados = sum(1 for tk, d, a, _, _ in candidatos if veredito(tk, d, a)[0])
        print(f"\n  {len(candidatos)} candidatos, {cortados} passariam os limiares atuais")
        print("  O piso de cada token e o maior entre `filler_min_seconds` e")
        print(f"  {rules.filler_stretch_ratio:g}x a mediana daquele token nesta fala"
              " — e o que separa verbo")
        print("  abrindo frase de hesitacao alongada.")

        # A mediana por token e a calibracao que sai da propria fala, entao
        # vale ver os numeros: e com eles que se escolhe o stretch_ratio.
        ambiguos = [(tk, medians[tk]) for tk in sorted(medians)
                    if tk in wanted and tk in {"e", "a", "o", "um", "uma"}]
        if ambiguos:
            print("\n  Mediana dos tokens ambiguos nesta fala "
                  "(base do piso relativo):")
            print("  " + "  ".join(f"{tk}={md:.2f}s" for tk, md in ambiguos))

    # --- o que o plano atual faz ------------------------------------------
    plano = build_plan(transcript, rules, FPS)
    s = plano.stats
    print(f"\n  Com os limiares atuais do config:")
    print(f"  {s.n_cuts} cortes ({s.n_pause_cuts} pausas, {s.n_filler_cuts} hesitacoes), "
          f"{s.removed_seconds:.1f}s removidos ({s.removed_ratio:.1%})\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    main(Path(sys.argv[1]))
