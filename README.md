# VicCount

Pui o poză și îți spune câte obiecte sunt. Merge pe **telefon**, totul rulează local, fără server. După prima deschidere funcționează și **offline** (inclusiv în modul avion).

**Aplicația:** https://victorastulum065.github.io/VicCount/ → din meniul browserului: *Adaugă pe ecranul principal*.

## Cum numără

| Ce faci | Ce se întâmplă |
|---|---|
| Doar poza | automat: găsește tipul dominant de obiect și le numără |
| Scrii ce (ro sau en: *sticle, oameni, monede*) | numără doar asta |
| Tragi un chenar peste **un** obiect | numără tot ce seamănă cu el (cel mai precis, merge pe orice) |
| Atingi un punct / un loc gol | scoți / adaugi manual |

Modele, rulate în browser (ONNX, int8):
- **Grounding DINO tiny**: detecție open-vocabulary după text.
- **DINOv2 (cu registre) small**: numărare după exemplu prin similaritate de trăsături, metoda *CountingDINO* (WACV 2026, fără antrenare, pe orice obiect).

Pentru puține obiecte se folosește detectorul. Pentru multe obiecte sau după exemplu se folosește similaritatea DINOv2, pe bucăți de înaltă rezoluție.

Prima deschidere descarcă ~250 MB (model + motor), o singură dată. De preferat pe Wi-Fi.

## Publicare

`.github/workflows/pages.yml` construiește site-ul la fiecare push pe `main`: copiază `phone/`, ia transformers.js + ONNX Runtime din npm și modelele de pe Hugging Face. Totul e servit de pe GitHub Pages, de pe același domeniu.

Setare unică: **Settings → Pages → Source: GitHub Actions**.

## Varianta PC (mai precisă)

`pc/`: server local cu **CountGD++** (CVPR 2026), accesibil și de pe telefon în rețeaua locală. Vezi `pc/README.md`.
