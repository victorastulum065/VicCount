# VicCount

Pui o poză → îți spune câte obiecte sunt. Orice obiecte. Rulează **100% local, offline**, din browser (PC sau telefon în aceeași rețea).

Model: **CountGD++** (Amini-Naieni & Zisserman, CVPR 2026, licență MIT) — numărare open-world: nu e antrenat pe clase fixe, numără orice îi arăți.

## Instalare (o singură dată, cu internet)

Necesită Python 3.10+. Descarcă ~1.7 GB (model + BERT).

| Sistem | Comandă |
|---|---|
| Windows | dublu-click `setup.bat` |
| macOS / Linux | `./setup.sh` |

Pe PC cu placă NVIDIA se instalează automat varianta CUDA.

## Pornire (offline)

- Windows: `start.bat`
- macOS / Linux: `./start.sh`

Deschide `http://localhost:8765`. Din telefon: adresa afișată în consolă (`http://IP-PC:8765`).

## Utilizare

| Cum | Ce face |
|---|---|
| Doar poza | **Automat** — numără tipul de obiect dominant din poză |
| Poza + text (engleză: `bottle`, `person`, `coin`) | numără doar ce ai scris |
| Poza + chenar tras peste un obiect | numără tot ce seamănă cu exemplul (cel mai precis) |
| Câmpul *Exclude* | ce să **nu** numere (ex: `cap` la sticle) |

- **Sensibilitate** — recalculează live, fără să ruleze iar modelul.
- **Tap pe un punct** — îl scoate/pune la loc (corecție manuală).
- **Salvează PNG** — poza adnotată cu totalul.

Automat, VicCount face o a 2-a trecere cu „pseudo-exemplu" (cel mai sigur obiect găsit devine exemplu vizual — metoda din lucrarea CountGD++), iar la poze foarte aglomerate (peste ~700 obiecte) împarte imaginea în bucăți și unește rezultatele.

## API

```
POST /api/count   (multipart)
  image     fișier
  text      opțional
  neg_text  opțional
  boxes     opțional, JSON [[x1,y1,x2,y2], ...] în pixeli
  refine    true/false (a 2-a trecere)
→ {"count": 42, "boxes": [[cx,cy,w,h], ...], "scores": [...], "mode": "...", "seconds": 1.2}
```

```bash
curl -F image=@poza.jpg -F text=bottle http://localhost:8765/api/count
```

## Setări (variabile de mediu)

| Variabilă | Implicit | |
|---|---|---|
| `VICCOUNT_DEVICE` | `auto` | `cuda`, `cpu`, `mps` |
| `VICCOUNT_THREADS` | toate | fire CPU |
| `VICCOUNT_MAX_SIDE` | `2400` | latura maximă a imaginii |

Port: `start.bat --port 9000`.

## Viteză orientativă

GPU NVIDIA: sub 1 s / poză. CPU modern: ~5–15 s / trecere (modul automat face 2 treceri).

## Structură

```
server.py            server web + API
engine.py            logica de numărare (moduri, pseudo-exemplu, tiling)
download_weights.py  descărcare unică model
web/index.html       interfața
countgd/             cod model CountGD++ (MIT, vendorizat, rulează pe CPU fără compilare)
```

## Credite

CountGD++ — https://github.com/niki-amini-naieni/CountGDPlusPlus (MIT) · bazat pe GroundingDINO (Apache 2.0).
