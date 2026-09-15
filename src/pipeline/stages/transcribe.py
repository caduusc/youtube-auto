"""Estagio 2: faster-whisper local, em CPU, com timestamp por palavra."""

from __future__ import annotations

from ..config import Config
from ..log import log, stage
from ..schemas import Manifest, Transcript, TranscriptSegment, Word
from ..util import read_json_if_fresh, write_json


def run(manifest: Manifest, config: Config) -> Transcript:
    work = config.work_dir / manifest.slug
    transcript_path = work / "transcript.json"

    cached = read_json_if_fresh(transcript_path, Transcript, manifest.audio_hash)
    if cached is not None:
        log("transcribe.cached", slug=manifest.slug, segments=len(cached.segments))
        return cached

    with stage("transcribe", model=config.transcribe.model, device=config.transcribe.device):
        # import local: arrasta o CTranslate2, que nao e necessario nos outros
        # estagios nem na suite de testes
        from faster_whisper import WhisperModel

        model = WhisperModel(
            config.transcribe.model,
            device=config.transcribe.device,
            compute_type=config.transcribe.compute_type,
        )
        raw_segments, info = model.transcribe(
            str(config.root / manifest.audio_path),
            language=config.transcribe.language,
            beam_size=config.transcribe.beam_size,
            word_timestamps=True,
            vad_filter=config.transcribe.vad_filter,
        )

        segments: list[TranscriptSegment] = []
        for index, segment in enumerate(raw_segments):
            segments.append(TranscriptSegment(
                id=index,
                start=round(segment.start, 3),
                end=round(segment.end, 3),
                text=segment.text.strip(),
                words=[
                    Word(start=round(w.start, 3), end=round(w.end, 3), word=w.word)
                    for w in (segment.words or [])
                ],
            ))
            if index % 25 == 0:
                log("transcribe.progress", segment=index, at=f"{segment.end:.0f}s")

        if not segments:
            raise RuntimeError("a transcricao saiu vazia — confira o audio extraido")

        transcript = Transcript(
            input_hash=manifest.audio_hash,
            language=info.language,
            model=config.transcribe.model,
            duration=manifest.duration,
            segments=segments,
        )
        write_json(transcript_path, transcript)
        log("transcribe.ok", segments=len(segments),
            words=sum(len(s.words) for s in segments), language=info.language)
        return transcript
