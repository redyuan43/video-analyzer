# Diarization Benchmark - 2026-07-25

## Scope

- Providers: native 3D-Speaker and `pyannote/speaker-diarization-community-1`.
- Inputs: two real uploaded recordings already present on the AI device.
- Runtime: 3D-Speaker uses the existing native runtime; community-1 uses the isolated `/home/ai/pyannote-community-venv` on the compatible V100.
- This is an operational comparison, not a DER evaluation. No hand-labelled speaker reference is available for these recordings.

## Results

| Recording | Expected count source | Mode | Provider | Speakers | Turns | Speech seconds | Elapsed |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 4m32 real upload | Existing pipeline result: 2 | Automatic | 3D-Speaker | 2 | 72 | 176.300 | 45.691s |
| 4m32 real upload | Existing pipeline result: 2 | Automatic | community-1 | 2 | 94 | 107.781 | 28.452s |
| 16m55 real upload | Existing pipeline result: 5 | Automatic | 3D-Speaker | 7 | 192 | 433.480 | 51.291s |
| 16m55 real upload | Existing pipeline result: 5 | Automatic | community-1 | 12 | 224 | 261.069 | 48.192s |
| 16m55 real upload | Existing pipeline result: 5 | Constrained to 5 | 3D-Speaker | 5 | 188 | 433.480 | 50.763s |
| 16m55 real upload | Existing pipeline result: 5 | Constrained to 5 | community-1 | 5 | 208 | 261.069 | 47.794s |

## Assessment

1. Both providers reproduce the two-speaker count on the short recording.
2. On the longer recording, unconstrained community-1 over-clusters much more severely than 3D-Speaker: 12 speakers versus 7, against the existing five-speaker result.
3. Supplying the known speaker count stabilizes both providers. This control must be optional because many external recordings do not have a trusted count.
4. community-1 is faster on the V100 in these runs, but its voiced duration is substantially lower on both recordings. This requires manual timestamp review before treating it as more accurate.
5. Current recommendation: retain 3D-Speaker as the FireRed production diarization baseline. Keep community-1 as an AB provider until a small human-labelled Chinese meeting set can produce DER, missed-speech, and speaker-confusion measurements.

## Artifacts

- `/tmp/diarization-benchmark/real-upload-2speaker.3dspeaker.json`
- `/tmp/diarization-benchmark/real-upload-2speaker.pyannote-community.json`
- `/tmp/diarization-benchmark/real-upload-5speaker.3dspeaker.json`
- `/tmp/diarization-benchmark/real-upload-5speaker.pyannote-community.json`
- `/tmp/diarization-benchmark/real-upload-5speaker.3dspeaker-k5.json`
- `/tmp/diarization-benchmark/real-upload-5speaker.pyannote-community-k5.json`
