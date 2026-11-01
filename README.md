# VicCount

Pui o poză și îți spune câte obiecte sunt. Merge pe **telefon**, totul rulează local, fără server. După prima deschidere funcționează și **offline** (inclusiv în modul avion).

**Aplicația:** https://victorastulum065.github.io/VicCount/ → din meniul browserului: *Adaugă pe ecranul principal*.

## Cum numără

| Ce faci | Ce se întâmplă |
|---|---|
| Doar poza | automat: separă obiectele de fundal, alege singur un exemplu și numără tot ce seamănă |
| Tragi un chenar peste **un** obiect | numără tot ce seamănă cu el (cel mai precis, merge pe orice) |
| Atingi un punct / un loc gol | scoți / adaugi manual |

Model: **DINOv2 cu registre (small, int8, ~24 MB)**, rulat în browser (ONNX Runtime WASM). Numărarea după exemplu folosește metoda *CountingDINO* (WACV 2026, fără antrenare). Consum de memorie sub ~750 MB, deci merge pe telefon.

Automatul merge bine pe obiecte distincte pe fundal simplu. La obiecte lipite sau pe fundal încărcat, trage un chenar.

Prima deschidere descarcă ~45 MB, o singură dată.

## Publicare

`.github/workflows/pages.yml` construiește site-ul la fiecare push pe `main`: copiază `phone/`, ia transformers.js + ONNX Runtime din npm și modelele de pe Hugging Face. Totul e servit de pe GitHub Pages, de pe același domeniu.

Publicarea se face pe ramura `gh-pages` (fișierele >90 MB sunt tăiate în bucăți și lipite la loc de service worker). Dacă Pages nu pornește singur: Settings → Pages → Deploy from a branch → `gh-pages` / root.

## Varianta PC (mai precisă)

`pc/`: server local cu **CountGD++** (CVPR 2026), accesibil și de pe telefon în rețeaua locală. Vezi `pc/README.md`.
