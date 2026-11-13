# Training data for the 8-stem separator

All data lives in the `data\` folder of the project (`D:\raga-spatial\data`). It is not in git.

| Folder | What it is |
|---|---|
| `raw_stems\` | Tier 1 raw downloads (FreeSounds previews + YouTube audio) |
| `processed\stems\<category>\` | Tier 1 clips: 10 s, 16 kHz mono |
| `datasets\<name>\` | Open datasets, stems only (see below) |
| `processed\stems_t2\<category>\` | Tier 2 clips cut from those datasets |
| `models\` | Checkpoints (`separator_v3_best.pt` is what the app uses) |
| `logs\training_log_v3.json` | One line per epoch |

## Open datasets (Tier 2)

Downloaded by `backend\data_tools\download_datasets.py`, converted by `backend\data_tools\prepare_datasets.py`.
Every one is a direct, no-sign-up download from Zenodo.

| Name | Zenodo record | Download | Used for | Licence |
|---|---|---|---|---|
| Saraga Carnatic 1.5 | 4301737 | 14.4 GB | Carnatic vocals, violin, mridangam, ghatam (live, some bleed) | CC BY-NC-SA 4.0 |
| MUSDB18-HQ | 3338373 | 22.7 GB | vocals, bass, drums | Educational / non-commercial |
| URMP | 5034983 | 12.1 GB | solo bowed strings and wind instruments | CC0 |
| Medley-solos-DB | 3464194 | 7.9 GB | clarinet, flute, sax, trumpet, violin, piano, guitar, voice | CC BY 4.0 |
| GuitarSet | 3371780 | 0.7 GB | solo acoustic guitar | CC BY 4.0 |
| Mridangam Tani-avarthanam | 1266207 | 0.3 GB | solo mridangam | CC BY-NC-ND 4.0 |
| Slakh2100 (redux) | 4599666 | 104 GB | guitar, bass, piano, organ, synth, strings, brass, reeds, flutes, drums | CC BY 4.0 |

**Licence note.** Saraga, MUSDB18-HQ and the mridangam set are for
non-commercial use. A model trained on them inherits that limit: fine for
research, a portfolio or a demo, not for a paid product without replacing
that data or getting permission.

## Still thin

- `folk_texture` (dappu, parai, dhol, kombu) has no open dataset; it relies on
  Tier 1 clips only. It also overlaps with `percussion` by nature.
- Veena, nadaswaram, thavil, bansuri and harmonium have no open multitrack
  data. More clean solo recordings of these would help the most.

## Adding your own stems

Put 10 s, 16 kHz mono WAVs in `processed\stems_t2\<category>\`, named
`<set>__<source>__<number>.wav` (clips from the same recording share the
`<set>__<source>` part). Training picks them up at the start of its next epoch.
