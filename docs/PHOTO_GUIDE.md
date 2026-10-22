# Phone-photo tier: printing and shooting guide

The digital splits measure the readers on renders. The phone-photo tier measures them on real photos of **printed fictional documents** (DESIGN §9.1, §11 step 6). Every document is a SPECIMEN: never photograph a real ID for this.

## 1. Build the kit

```bash
uv run python tools/print_kit.py          # writes var/print/
```

You get one PDF per split (`var/print/<split>/print_<split>.pdf`, 37 A4 pages in all), `var/print/shotlist.csv` (112 rows) and `truth.json` per split.

| Split | Packets | Documents | Opened |
|---|---|---|---|
| dev | 4 | 16 | any time |
| calib | 8 | 32 | at certification |
| test_a | 4 | 16 | once, at the MVP (6 Nov) |
| test_b | 12 | 48 | once, at the progress review; one held-out layout per packet |

## 2. Print and cut

- Print at **100% / "actual size"**. Do not use "fit to page": the ID cards must come out at 85.6 × 54 mm. Check one with a ruler.
- **ID cards** (all `iq_national_card_*`, and card-sized chamber and tax documents): glossy photo paper if you can, cut on the marks, laminated, with rounded corners if possible.
- **Paper documents** (A5): plain card stock, cut on the marks.
- Keep each split's documents in its own envelope. A test-split photo must never be used for anything else.

## 3. Shoot

Follow `shotlist.csv` row by row: it says which phone (A or B) and which condition to use.

| Condition | How |
|---|---|
| normal | Daylight or room light, phone parallel to the document, document fills most of the frame |
| low light | A dim room, no flash |
| lamp glare | A desk lamp reflecting off the laminate |
| angle 30-45 deg | Tilt the phone noticeably |
| finger over a field | Hold the card so a finger covers the field named in `finger_over` |
| cut corner | Let one corner fall outside the frame |
| cluttered table | Papers, keys or a cable around the document |

- Save each photo under the name in `photo_file`, into `var/photos/<split>/`.
- Also send about 20 of them through WhatsApp to yourself and save the received copy as `<name>.whatsapp.jpg`.

## 4. After shooting

- For each photo, click the document's four corners: `uv run python tools/corner_labeler.py var/photos/<split>`, then open http://127.0.0.1:8770. Click top-left, top-right, bottom-right, bottom-left, as the document reads. The template's field boxes are projected through those corners, which gives ground-truth boxes for photos too.
- Fill `corners_clicked` in the shot list as you go.
- Test-A and Test-B photos are sealed (hashed) on 1 Nov, before anyone looks at results on them.
