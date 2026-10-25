# ✍️ Handwritten Digit Recognition (CNN + MNIST)

A complete, end-to-end web application that recognizes handwritten digits
(0–9). You draw a digit on a canvas (or upload an image), and a deep-learning
model instantly tells you which number it is — along with how confident it is
and a full breakdown of the probabilities for every digit.

**🌐 Live app:** https://digit-recognizer-ioi7.onrender.com
**💻 Source code:** https://github.com/Abhiram123897/handwritten-digit-recognition

> New to this project? Read the ["The Story"](#-the-story-what-we-built-and-how)
> section below — it walks through everything we did, in plain language, from
> zero to a live website.

---

## 📑 Table of Contents

1. [What this project does](#-what-this-project-does)
2. [The Story: what we built and how](#-the-story-what-we-built-and-how)
3. [The dataset (MNIST)](#-the-dataset-mnist)
4. [How the model was trained](#-how-the-model-was-trained)
5. [How the web app works](#-how-the-web-app-works)
6. [Project structure](#-project-structure)
7. [Architecture diagram](#-architecture-diagram)
8. [Run it yourself (step by step)](#-run-it-yourself-step-by-step)
9. [How we deployed it (Render)](#-how-we-deployed-it-render)
10. [Problems we hit and how we solved them](#-problems-we-hit-and-how-we-solved-them)
11. [API reference](#-api-reference)
12. [Results](#-results)
13. [Tech stack](#-tech-stack)
14. [Future enhancements](#-future-enhancements)

---

## 🎯 What this project does

- **Draw or upload** a handwritten digit (0–9).
- A **Convolutional Neural Network (CNN)** predicts the digit.
- You see the **predicted number**, a **confidence score**, and a **bar chart**
  of the probability for each digit 0–9.
- Every prediction is **saved to a history** you can view, export to CSV, or
  clear — all from a **dashboard** that also shows the model's training charts.
- Works on **desktop and mobile**, has a **dark mode**, and predicts in
  **real time** as you draw.

---

## 📖 The Story: what we built and how

Here is the whole journey, start to finish, in plain language.

### Step 1 — We decided what to build
The goal: a website where anyone can draw a digit and have an AI recognize it.
That needs three things working together:
1. A **trained AI model** that knows what digits look like.
2. A **backend server** that takes an image and runs the model.
3. A **web page (frontend)** where people draw and see results.

### Step 2 — We chose where it would eventually live (the cloud)
Before writing code we picked a hosting service, because that choice affects how
you build. We compared **Render**, **Vercel**, and **Netlify**:
- **Vercel / Netlify**: great for simple websites, but they can't run a
  long-lived Python AI server well (the AI library is too big, and they don't
  keep files like our history database between visits).
- **Render**: runs a real, always-available Python server — perfect for us.

So we chose **Render**, and built the app as a single Python (Flask) server.

### Step 3 — We trained the AI model
We used the famous **MNIST** dataset (see the next section) and built a **CNN**
(a type of neural network that's excellent at images). We trained it on 60,000
example images until it could recognize digits with **99.49% accuracy** on
10,000 test images it had never seen. Training produced the model file the app
uses to make predictions. (Full details in
["How the model was trained"](#-how-the-model-was-trained).)

### Step 4 — We built the backend (the server)
Using **Flask** (a Python web framework), we created a server that:
- loads the trained model once when it starts,
- accepts an image (from the drawing canvas or an uploaded file),
- cleans up the image so it looks like MNIST digits,
- runs the model and returns the answer,
- saves each prediction into a small **SQLite** database.

### Step 5 — We built the frontend (the web page)
Using **HTML, CSS, JavaScript, Bootstrap 5, and Chart.js**, we made a modern,
responsive page with a drawing canvas, image upload (with drag-and-drop), a
confidence bar, a probability chart, loading spinners, and a dark-mode toggle.
We also built a **dashboard** page showing the history table and the model's
training charts.

### Step 6 — We tested everything locally
We ran the app on a local machine, drew digits, uploaded images, checked the
dashboard, exported CSV, and confirmed predictions were correct before going
further.

### Step 7 — We put the code on GitHub
We created a Git repository and pushed all the code (and the trained model) to
GitHub, so it has a home online and can be deployed.

### Step 8 — We deployed it to Render (and made it live)
We connected the GitHub repo to Render using a "blueprint" file (`render.yaml`).
Render installed the dependencies and started the server. After fixing a couple
of real-world issues (see
["Problems we hit"](#-problems-we-hit-and-how-we-solved-them)), the app went
**live on the internet** at the URL above.

That's the entire project in a nutshell. The rest of this README goes deeper on
each part.

---

## 🔢 The dataset (MNIST)

We used **MNIST**, the most well-known dataset for handwritten digit
recognition.

- **What it is:** 70,000 grayscale images of handwritten digits (0–9).
- **Split:** 60,000 images for training, 10,000 for testing.
- **Image size:** 28×28 pixels, grayscale (white digit on black background).
- **Labels:** every image comes tagged with the correct digit.
- **Where it came from:** it's built into TensorFlow/Keras, so we load it with a
  single line of code — no manual download needed:
  ```python
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  ```

We used the 60,000 training images to teach the model, and kept the 10,000 test
images hidden during training so we could fairly measure how well it learned.

Credit: MNIST was created by Yann LeCun, Corinna Cortes, and Christopher Burges.

---

## 🧠 How the model was trained

All training happens in **`train.py`**. Here's what it does, in order:

1. **Load the data** — download MNIST (handled automatically by Keras).
2. **Preprocess** —
   - reshape each image to `28×28×1` (height × width × 1 color channel),
   - **normalize** pixel values from `0–255` down to `0–1` (helps the model
     learn faster and more stably),
   - **one-hot encode** the labels (e.g. the digit `3` becomes
     `[0,0,0,1,0,0,0,0,0,0]`).
3. **Build the CNN** — a Convolutional Neural Network with this shape:
   ```
   Input 28×28×1
   → Conv(32) → BatchNorm → Conv(32) → MaxPool → Dropout
   → Conv(64) → BatchNorm → Conv(64) → MaxPool → Dropout
   → Flatten → Dense(128) → Dropout → Dense(10, softmax)
   ```
   - **Conv layers** detect visual patterns (edges, curves, loops).
   - **MaxPool** shrinks the image while keeping important features.
   - **Dropout / BatchNorm** prevent over-fitting and stabilize training.
   - The final **softmax** layer outputs 10 probabilities (one per digit).
4. **Train** — the model studies the 60,000 training images for up to 15 passes
   ("epochs"), using the **Adam** optimizer. "Early stopping" halts training if
   it stops improving, and the learning rate is reduced automatically when
   progress plateaus.
5. **Evaluate** — test on the 10,000 unseen images. We reached
   **99.49% accuracy**.
6. **Save everything the app needs:**
   - `model/digit_model.h5` — the full trained model (Keras format).
   - `model/digit_model.tflite` — a tiny, fast version for serving (~469 KB).
   - `model/metrics.json` — accuracy, loss, and per-digit scores.
   - `static/images/accuracy_loss.png` — training charts.
   - `static/images/confusion_matrix.png` — where the model gets confused.

To train it yourself: `python train.py` (takes a few minutes on a laptop CPU).

> **Why two model files?** The `.h5` is the "real" trained model. The `.tflite`
> is a slimmed-down copy that runs predictions using far less memory — important
> for running cheaply in the cloud (more on that below).

---

## 🖥️ How the web app works

When you draw a digit and hit predict, here's the full round trip:

1. The browser turns your drawing into a **PNG image** (or you upload a file).
2. It sends that image to the server's **`/predict`** endpoint.
3. The server (**`app.py`**) uses **OpenCV** to clean the image so it matches
   MNIST style: convert to grayscale, make it white-on-black, crop tightly to
   the digit, and center it on a **28×28** canvas.
4. The model runs and returns **10 probabilities** (one per digit).
5. The server picks the highest one as the answer, saves the result to the
   **SQLite** database, and sends back JSON with the digit, confidence, and all
   10 probabilities.
6. The browser shows the big predicted number, fills the **confidence bar**, and
   draws the **probability chart**.

The **dashboard** page separately calls `/history` and `/metrics` to show your
past predictions and the model's performance.

---

## 🧱 Project structure

```
handwritten-digit-recognition/
├── app.py                   # Flask server: loads model, /predict, history, metrics
├── train.py                 # Trains the CNN and saves model + charts + metrics
├── requirements.txt         # Lightweight dependencies for RUNNING the app (server)
├── requirements-train.txt   # Heavier dependencies for TRAINING the model
├── render.yaml              # Instructions Render uses to deploy the app
├── README.md                # This file
├── PLAN.md                  # The original project plan and decisions
├── .gitignore
│
├── model/
│   ├── digit_model.h5       # Trained Keras model  (created by train.py)
│   ├── digit_model.tflite   # Lightweight serving model (created by train.py)
│   └── metrics.json         # Performance numbers (created by train.py)
│
├── database/
│   └── history.db           # SQLite prediction history (auto-created at runtime)
│
├── static/
│   ├── css/style.css        # Styling: gradients, cards, dark mode, responsive
│   ├── js/script.js         # Canvas drawing, upload, chart, dark mode
│   ├── js/dashboard.js      # History table, metrics, CSV export, clear
│   └── images/              # accuracy_loss.png, confusion_matrix.png (from training)
│
├── templates/
│   ├── index.html           # Main page: draw / upload / see prediction
│   └── dashboard.html       # Dashboard: history + charts + metrics
│
└── uploads/                 # Saved thumbnails of predicted images
```

---

## 🏗️ Architecture diagram

```
            ┌──────────────────────────────────────────────┐
            │                  Browser (UI)                  │
            │  index.html / dashboard.html + Bootstrap 5     │
            │  Draw canvas · Upload · Chart.js · Dark mode   │
            └───────────────┬───────────────▲───────────────┘
                 send image (POST /predict)  │  JSON result
                            ▼                │  (digit, confidence, probs)
            ┌──────────────────────────────────────────────┐
            │                 Flask server (app.py)          │
            │   routes: / /dashboard /predict /history ...   │
            │   OpenCV cleans image → 28×28 MNIST style      │
            │   Model runs via lightweight TFLite runtime    │
            └──────┬───────────────────────┬────────────────┘
                   ▼                        ▼
         ┌──────────────────┐     ┌────────────────────────┐
         │ model/*.tflite   │     │ database/history.db      │
         │ (the trained AI) │     │ uploads/*.png (thumbs)   │
         └──────────────────┘     └────────────────────────┘
                   ▲
                   │ produced by
         ┌──────────────────┐
         │     train.py     │   MNIST → CNN → train → evaluate → save
         └──────────────────┘
```

---

## ▶️ Run it yourself (step by step)

**Requirements:** Python **3.10–3.12** (TensorFlow does not yet support 3.13+).

```bash
# 1. Get the code
git clone https://github.com/Abhiram123897/handwritten-digit-recognition.git
cd handwritten-digit-recognition

# 2. Create a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

**3. Install dependencies** — there are two files:

```bash
# To RUN the app only (lightweight — the trained model is already included):
pip install -r requirements.txt

# To also TRAIN the model yourself (heavier — adds TensorFlow etc.):
pip install -r requirements-train.txt
```

**4. (Optional) Train the model** — only if you want to retrain. A trained model
is already in `model/`, so you can skip this:

```bash
python train.py
```

**5. Start the app:**

```bash
python app.py
# then open http://localhost:5000
```

Open **http://localhost:5000** for the recognizer and
**http://localhost:5000/dashboard** for the dashboard.

> **Tip (Windows):** make sure your virtual environment uses Python 3.12, e.g.
> `py -3.12 -m venv venv`. If `python --version` shows 3.13 or 3.14, install
> Python 3.12 alongside it — TensorFlow has no builds for 3.13/3.14 yet.

---

## ☁️ How we deployed it (Render)

We deployed the live site to **Render**. Here's exactly how, so you can
reproduce it:

1. **Push the project to GitHub** (done — see the repo link at the top).
2. Sign in to **https://render.com** using your **GitHub** account.
3. Click **New + → Blueprint**.
4. Grant Render access to the repository, then select
   **handwritten-digit-recognition**.
5. Render reads **`render.yaml`** automatically. It defines one free web service
   that:
   - installs dependencies with `pip install -r requirements.txt`,
   - starts the server with `gunicorn app:app`,
   - checks health at `/health`.
6. Give the blueprint a name and click **Apply / Deploy**.
7. Wait for the build to finish and the status to turn **Live**. Render gives you
   a public URL (ours is https://digit-recognizer-ioi7.onrender.com).

Every time you `git push` to the `main` branch, Render **auto-deploys** the new
version.

> **Free-tier notes:** the free service "sleeps" after ~15 minutes of
> inactivity, so the **first visit** after a pause can take ~50 seconds to wake
> up (later requests are fast). Also, the free tier has no permanent disk, so the
> prediction **history resets** whenever the server restarts. Both are fine for a
> demo/portfolio app.

---

## 🧯 Problems we hit and how we solved them

Real projects hit snags. Here are the ones we solved, kept here so the learning
is visible:

1. **Python version was too new.**
   The machine had Python 3.14, but TensorFlow has no builds for it. **Fix:**
   installed Python 3.12 alongside it and created the virtual environment with
   `py -3.12`.

2. **A corrupted download broke `pip install`.**
   An interrupted download caused a "hashes do not match" error. **Fix:**
   `pip cache purge` then reinstalling with `--no-cache-dir`.

3. **Render rejected a persistent disk on the free tier.**
   Our first `render.yaml` asked for a storage disk, which free services don't
   allow. **Fix:** removed the disk (history now resets on restart, acceptable
   for a free demo).

4. **The app crashed on the free tier with a 502 error.**
   The biggest one. Loading the **full TensorFlow** library to make predictions
   used more than Render's 512 MB memory limit, so the server got killed on
   every prediction. **Fix:** switched the running app to a tiny standalone
   prediction engine (**`ai-edge-litert`**) that loads the `.tflite` model using
   a fraction of the memory. We split dependencies into two files —
   `requirements.txt` (lightweight, for the server) and `requirements-train.txt`
   (heavy, only for training). After this, the app ran comfortably within the
   free-tier memory and went live. 

---

## 🔌 API reference

| Method | Endpoint          | Description |
|--------|-------------------|-------------|
| GET    | `/`               | Main page (draw / upload / predict) |
| GET    | `/dashboard`      | Dashboard page |
| POST   | `/predict`        | Predict from canvas (JSON `{image}`) or upload (form `image`) |
| GET    | `/history`        | Prediction history (JSON) |
| POST   | `/history/clear`  | Clear all history |
| GET    | `/history/export` | Download history as CSV |
| GET    | `/metrics`        | Model performance metrics (JSON) |
| GET    | `/health`         | Health check (used by Render) |

**Example `/predict` response:**
```json
{
  "digit": 7,
  "confidence": 0.9987,
  "probabilities": [0.0001, 0.0, "...", 0.9987, "..."],
  "image_url": "/uploads/pred_20261024_171500_000000.png",
  "timestamp": "2026-10-24 17:15:00"
}
```

---

## 📊 Results

- **Test accuracy: 99.49%** (on 10,000 unseen images)
- **Test loss: 0.0164**
- **Epochs trained: 15**
- **Per-digit F1 scores:** all between **0.9927 and 0.998** — the model is
  strong and consistent across every digit.

Training charts are generated into `static/images/` and shown on the dashboard:
- `accuracy_loss.png` — accuracy and loss over training.
- `confusion_matrix.png` — which digits get mistaken for which (very few do).

---

## 🧰 Tech stack

**Machine learning:** Python · TensorFlow/Keras · NumPy · scikit-learn ·
Matplotlib/Seaborn (training only)
**Serving:** ai-edge-litert (lightweight TFLite runtime) · OpenCV · Pillow
**Backend:** Flask · SQLite · Gunicorn
**Frontend:** HTML5 · CSS3 · JavaScript · Bootstrap 5 · Chart.js
**Deployment:** GitHub · Render

---

## 🔮 Future enhancements

- Recognize **multiple digits** in one image (segment and read a sequence).
- Extend to **letters** (EMNIST) or custom symbols.
- **User accounts** with per-user history.
- A **paid persistent database** so history survives restarts.
- True streaming **real-time** prediction via WebSockets.
- A **Dockerfile** and CI/CD pipeline for one-command deploys anywhere.

---

## ✅ Quick local setup (copy-paste, step by step)

The exact steps to run this on your own machine from scratch. Windows commands
are shown first; macOS/Linux equivalents are noted where they differ.

**Step 1 — Check your Python version.**
```powershell
python --version
```
You need **Python 3.10, 3.11, or 3.12**. TensorFlow has **no** builds for 3.13
or 3.14 yet, so if you see those, do Step 1a.

**Step 1a — (Only if your Python is 3.13/3.14) install Python 3.12 alongside it.**
```powershell
winget install Python.Python.3.12
# then close & reopen the terminal, and confirm it is detected:
py -0
```
Your existing Python stays untouched — 3.12 installs side by side.

**Step 2 — Get the code.**
```powershell
git clone https://github.com/Abhiram123897/handwritten-digit-recognition.git
cd handwritten-digit-recognition
```

**Step 3 — Create and activate a virtual environment (using Python 3.12).**
```powershell
# Windows (use py -3.12 to force the right version):
py -3.12 -m venv venv
venv\Scripts\activate

# macOS/Linux:
# python3.12 -m venv venv
# source venv/bin/activate
```
Confirm the venv is on 3.12:
```powershell
python --version      # should print Python 3.12.x
```

**Step 4 — Install dependencies.**
```powershell
# To just RUN the app (lightweight; the trained model is already included):
pip install -r requirements.txt

# OR, to also RETRAIN the model yourself (adds TensorFlow etc.):
pip install -r requirements-train.txt
```
> If you ever get a "hashes do not match" error, a download was interrupted.
> Fix it with:
> ```powershell
> pip cache purge
> pip install --no-cache-dir -r requirements.txt
> ```

**Step 5 — (Optional) Retrain the model.**
A trained model already ships in `model/`, so you can skip this. To retrain:
```powershell
python train.py
```
(Only works if you installed `requirements-train.txt`. Takes a few minutes.)

**Step 6 — Start the app.**
```powershell
python app.py
```
You should see `Running on http://127.0.0.1:5000`.

**Step 7 — Open it in your browser.**
- Recognizer:  http://localhost:5000
- Dashboard:   http://localhost:5000/dashboard

Draw a digit, hit Predict, and you're done. To stop the server, press
**Ctrl + C** in the terminal.

---

## 📝 License & acknowledgements

MNIST dataset by Yann LeCun, Corinna Cortes, and Christopher Burges.
Educational / portfolio project — free to use and adapt.
