# TSHM — ESP32 Demo (README)

**Short:** This repository demonstrates *runtime behavior* of the TSHM streaming model on an ESP32 device: real-time streaming **inference** and **on-device head-only learning (SGD)**.
**Important:** The `builds/hello_world.bin` included in Releases is a **demonstration binary** showing runtime metrics (latency, memory pattern, on-device SGD time) using *example/exported* weights. The *true/full* weight files (production or unquantized FP32 weights) are **not** included in the public binary — they can be provided privately on request (see “Requesting true weights” below).

> Why? many companies or projects want to publish a reproducible runtime demo but keep trained weights private or distribute them on request. This README explains how to reproduce the runtime, what the demo binary shows, and how to request the full weights.

---

## Repo layout (recommended)

```
/
├─ firmware/                       # ESP-IDF project (main.cpp or tshm_main.cpp, components/, CMakeLists)
│  ├─ main/
│  │  ├─ main.cpp                   # device demo (I2S/mfcc/tshm)
│  ├─ components/
│  │  ├─ tshm/                      # component with tshm_weights.hpp template
│  ├─ CMakeLists.txt
│  └─ sdkconfig
├─ assets/                          # example MFCC headers or small audio assets
├─ python/                          # training / export tools and parse_serial.py
│  ├─ export_weights.py
│  └─ parse_serial.py
├─ builds/                          # optional: prebuilt demo binary (hello_world.bin)
│  └─ hello_world.bin
├─ docs/
│  └─ results.md                    # measured metrics & reproducibility notes
├─ README.md                        # (this file)
└─ LICENSE
```

---

# DEMO BINARY DISCLAIMER (IMPORTANT)

The `builds/hello_world.bin` attached to the GitHub Release is provided **only** to demonstrate:

* **Per-frame inference runtime** on a target ESP32 (observed log lines like `Frame time: 1186 us`).
* **On-device head-only learning runtime** (observed log lines like `Head SGD total time: 1217 us` for 10 updates).
* Typical memory allocation pattern (PSRAM/internal RAM usage and scratch buffer scheme).
* End-to-end demo flow (embedding → streaming forward_step → head logits → optional on-device head SGD).

**It is NOT intended to be the production weight file**. If you want the original (FP32) exported weights, or the training checkpoints that produced the weights, I will provide them on request (see below). The demo binary uses exported assets/weights chosen to make the demo self-contained and reproducible.

---

## Quick reproducibility facts (from demo runs)

* **Model params (reported):** ~**7,500** parameters
* **Weights stored (quantized INT8):** ~**~7.5 KB** (INT8) — or **~29 KB** if stored as full FP32 arrays (approx; depends on export).
* **Per-frame inference latency (ESP32-S3 example):** typical log lines show **~1.1–1.4 ms** per frame (example `Asset Frame X time: 1186 us`).
* **Head-only on-device SGD:** example demo applied 10 SGD updates in **~1,217 µs total** → ~**122 µs per update** (very cheap because only the head is updated).
* **Dataset accuracy (example training):** e.g. **~0.876** on SC-10 evaluation (your training results; include your exact run details in `docs/results.md`).

> Put your exact measured numbers and environment (chip, IDF version, compile flags) into `docs/results.md` for reproducibility.

---

## Build & flash (reproduce demo locally)

### Prerequisites

* ESP-IDF v5.x (or the version used to build the published binary)
* Toolchain & Python 3.8+
* USB cable to the target device (ESP32-S3 recommended for demo)

### Quick build steps

```bash
# clone
git clone https://github.com/<you>/tshm-esp-demo.git
cd tshm-esp-demo/firmware

# source esp-idf environment (example)
. $IDF_PATH/export.sh

# optionally generate tshm_weights.hpp (if you want to rebuild weights locally)
python3 ../python/export_weights.py --out components/tshm/include/tshm_weights.hpp --quantize int8

# set target and build
idf.py set-target esp32s3
idf.py build

# flash & monitor
idf.py -p /dev/ttyUSB0 flash monitor
```

### Use prebuilt demo binary

* Download `builds/hello_world.bin` from the Release page.
* Either instruct users to unpack it into an ESP-IDF project or provide an esptool flash command (if you include offset details). The safest route for newcomers is to provide an IDF project + `hello_world.bin` placed into `firmware/build/` and then `idf.py -p /dev/ttyUSB0 flash monitor`.

---

## Interpreting demo logs — what is inference and what is learning?

Example log lines from the demo:

```
I (548) TSHM_DEVICE: Predicted class BEFORE_SGD ... (logit=19.144442)
I (558) TSHM_DEVICE: Head SGD total time: 1217 us
I (608) TSHM_DEVICE: Asset Frame 2/161 time: 1190 us | Predicted id=0 logit=19.125647
```

* `Asset Frame X/Y time: <us>` or `Frame time: <us>` → **per-frame inference latency** (this is the time for `forward_step` + logits). This is the principal runtime metric for real-time classification.
* `Head SGD total time: <us>` → **on-device learning runtime** for the demo training run; if demo applied N updates, per-update time = `total / N`.
* MFCC extraction (if done on-device) is a separate cost — the demo may use precomputed MFCCs or a light MFCC pipeline; include MFCC timing in `docs/results.md` if you run MFCC on-device.

**So:** the `~1.2 ms` log is inference. The SGD lines refer only to the tiny head updates and are independent (and much cheaper unless you update many times).

---

## Tips to make the demo more convincing for product / resume

1. **Docs/results.md** — include:

   * exact device model, IDF version, CPU frequency, compiler flags
   * average frame latency ± stdev, head SGD per-update time, max heap usage, flash usage
   * confusion matrix and accuracy numbers for the dataset used
   * the exact commands used to measure & parse logs

2. **Provide both quantized INT8 and full FP32 exports** (export both `tshm_weights.hpp` forms). The demo binary can use INT8; offer FP32 on request for people who need the original weights.

3. **Add a `parse_serial.py`** script (included in repo) to parse logs and produce mean/median/min/max/stdev of frame times — remove ambiguity and make claims reproducible.

4. **Publish a Release** with:

   * `hello_world.bin` (demo)
   * `serial.log` capture showing the lines you claim
   * `docs/results.md` with measurement methodology

5. **Repro project**: include a `Makefile` or `idf.py` build script that points to the exact `sdkconfig` and build flags you used.

6. **Resume/Company sell points**: put measured metrics (params, frame latency, on-device learning cost, flash size) and link to the GitHub demo, plus a short note: “weights available on request”.

---

## How to request the true/full weights

If you (or a company) want the original (FP32) weights or the training checkpoint:

* **Preferred:** open a GitHub Issue in this repo with the subject `Request: FP32 weights / training checkpoint` and provide your contact email or GitHub account; I will reply with a private link or instructions.
* **Alternative:** send an email to `<your-email-here>` (replace with desired contact) — if you want me to prepare a private transfer, mention the project and intended use.

(If you are the repo owner, change this section to the contact method you prefer — e.g., direct download link on request, or add weights to Releases under a restricted access method.)

---

## Example `parse_serial.py` (already available in `python/`)

Use the included parser to extract frame & SGD times and compute statistics. Example usage:

```bash
idf.py -p /dev/ttyUSB0 monitor > serial.log
python3 python/parse_serial.py serial.log
```

This removes ambiguity over whether the printed microseconds are per-frame or per-N-frames.

---

## License & attribution

* Include a LICENSE (MIT recommended) and a short contributor note if needed.
* If you derived TSHM from prior research or others’ code, add an acknowledgements section.


