import json

with open('rating_data.json', encoding='utf-8') as f:
    rows = json.load(f)

data_json = json.dumps(rows, ensure_ascii=False).replace('</script', '<\\/script')

TEMPLATE = r'''<title>PidginEdu Rating Study</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Karla:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #F7F5EF;
    --surface: #FFFFFF;
    --surface-raised: #FFFFFF;
    --ink: #1D2420;
    --ink-soft: #5B655D;
    --ink-faint: #93998F;
    --border: #E2DED2;
    --border-soft: #ECE8DC;
    --accent: #2C6E4E;
    --accent-ink: #FFFFFF;
    --accent-soft: #E4EEE7;
    --warm: #B8712C;
    --warm-soft: #F3E7D6;
    --danger: #A5432D;
    --shadow: 0 1px 2px rgba(29,36,32,0.04), 0 8px 24px rgba(29,36,32,0.06);
    --radius: 14px;
    --serif: "Fraunces", Georgia, "Times New Roman", serif;
    --sans: "Karla", "Segoe UI", system-ui, -apple-system, sans-serif;
  }

  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #14180F;
      --surface: #1C2116;
      --surface-raised: #232A1C;
      --ink: #EDEBE0;
      --ink-soft: #A6AC9B;
      --ink-faint: #6E7566;
      --border: #333B29;
      --border-soft: #2A311F;
      --accent: #59A97E;
      --accent-ink: #0D1409;
      --accent-soft: #223026;
      --warm: #DE9A4E;
      --warm-soft: #3A2E1B;
      --danger: #D97B5F;
      --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 8px 24px rgba(0,0,0,0.35);
    }
  }
  :root[data-theme="dark"] {
    --bg: #14180F;
    --surface: #1C2116;
    --surface-raised: #232A1C;
    --ink: #EDEBE0;
    --ink-soft: #A6AC9B;
    --ink-faint: #6E7566;
    --border: #333B29;
    --border-soft: #2A311F;
    --accent: #59A97E;
    --accent-ink: #0D1409;
    --accent-soft: #223026;
    --warm: #DE9A4E;
    --warm-soft: #3A2E1B;
    --danger: #D97B5F;
    --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 8px 24px rgba(0,0,0,0.35);
  }

  * { box-sizing: border-box; }
  html { -webkit-text-size-adjust: 100%; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--ink);
    font-family: var(--sans);
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
  }
  @media (prefers-reduced-motion: reduce) {
    * { animation-duration: 0.001ms !important; transition-duration: 0.001ms !important; }
  }

  .shell {
    max-width: 760px;
    margin: 0 auto;
    padding: 28px 20px 80px;
  }

  header.top {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 28px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border-soft);
  }
  header.top .name {
    font-family: var(--serif);
    font-weight: 600;
    font-size: 1.15rem;
    letter-spacing: -0.01em;
  }
  header.top .rater {
    font-size: 0.82rem;
    color: var(--ink-soft);
    font-variant-numeric: tabular-nums;
  }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
  }

  /* ---- intro / name screens ---- */
  .intro { padding: 40px 36px; }
  .intro .eyebrow {
    text-transform: uppercase;
    letter-spacing: 0.09em;
    font-size: 0.72rem;
    color: var(--accent);
    font-weight: 700;
    margin: 0 0 10px;
  }
  .intro h1 {
    font-family: var(--serif);
    font-size: clamp(1.6rem, 4vw, 2.1rem);
    font-weight: 600;
    letter-spacing: -0.015em;
    text-wrap: balance;
    margin: 0 0 18px;
  }
  .intro p { color: var(--ink-soft); font-size: 0.98rem; margin: 0 0 14px; max-width: 58ch; }
  .intro ul { color: var(--ink-soft); font-size: 0.98rem; padding-left: 1.2em; margin: 0 0 14px; }
  .intro li { margin-bottom: 6px; }
  .intro strong { color: var(--ink); }

  label.field-label {
    display: block;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--ink-soft);
    margin-bottom: 8px;
  }
  input[type="text"] {
    width: 100%;
    font-family: var(--sans);
    font-size: 1.05rem;
    padding: 12px 14px;
    border-radius: 10px;
    border: 1px solid var(--border);
    background: var(--bg);
    color: var(--ink);
  }
  input[type="text"]:focus, button:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
  }

  button {
    font-family: var(--sans);
    cursor: pointer;
    border: none;
  }
  .btn-primary {
    background: var(--accent);
    color: var(--accent-ink);
    font-weight: 700;
    font-size: 1rem;
    padding: 13px 26px;
    border-radius: 10px;
    transition: opacity 0.15s ease;
  }
  .btn-primary:hover { opacity: 0.88; }
  .btn-primary:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn-ghost {
    background: transparent;
    color: var(--ink-soft);
    font-size: 0.88rem;
    padding: 8px 4px;
    text-decoration: underline;
    text-underline-offset: 3px;
  }

  /* ---- progress ---- */
  .progress-wrap { margin-bottom: 22px; }
  .progress-meta {
    display: flex;
    justify-content: space-between;
    font-size: 0.78rem;
    color: var(--ink-soft);
    font-variant-numeric: tabular-nums;
    margin-bottom: 7px;
  }
  .progress-track {
    height: 5px;
    background: var(--border-soft);
    border-radius: 999px;
    overflow: hidden;
  }
  .progress-fill {
    height: 100%;
    background: var(--accent);
    border-radius: 999px;
    transition: width 0.25s ease;
  }

  /* ---- rating item ---- */
  .item-card { padding: 26px 26px 24px; }
  .subject-tag {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--warm);
    background: var(--warm-soft);
    padding: 3px 9px;
    border-radius: 999px;
    margin-bottom: 14px;
  }
  .context-block { margin-bottom: 20px; }
  .context-row { margin-bottom: 10px; }
  .context-row .k {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--ink-faint);
    margin-bottom: 3px;
  }
  .context-row .v {
    font-size: 1rem;
    color: var(--ink-soft);
  }
  .context-row.reference .v { color: var(--ink); }

  .options {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin: 18px 0 20px;
  }
  @media (max-width: 560px) {
    .options { grid-template-columns: 1fr; }
  }
  .option {
    border: 1.5px solid var(--border);
    border-radius: 12px;
    padding: 16px 16px 14px;
    background: var(--bg);
    cursor: pointer;
    text-align: left;
    display: flex;
    flex-direction: column;
    gap: 8px;
    transition: border-color 0.12s ease, background 0.12s ease;
  }
  .option:hover { border-color: var(--accent); }
  .option .opt-label {
    font-family: var(--serif);
    font-weight: 600;
    font-size: 0.95rem;
    color: var(--ink-soft);
  }
  .option .opt-text {
    font-size: 1.02rem;
    color: var(--ink);
    line-height: 1.55;
  }
  .option.selected {
    border-color: var(--accent);
    background: var(--accent-soft);
  }
  .option.selected .opt-label { color: var(--accent); }
  .option .opt-check {
    display: none;
    font-size: 0.78rem;
    font-weight: 700;
    color: var(--accent);
  }
  .option.selected .opt-check { display: block; }

  .identical-note {
    text-align: center;
    font-size: 0.85rem;
    color: var(--warm);
    background: var(--warm-soft);
    border-radius: 8px;
    padding: 10px 14px;
    margin: -6px 0 16px;
  }

  mark.diffw {
    background: var(--warm-soft);
    color: var(--ink);
    box-shadow: 0 0 0 1px var(--warm);
    border-radius: 3px;
    padding: 0 2px;
    font-style: normal;
  }

  .tie-row { text-align: center; margin: -8px 0 20px; }
  .tie-btn {
    background: transparent;
    border: 1px dashed var(--border);
    color: var(--ink-soft);
    font-size: 0.85rem;
    padding: 8px 18px;
    border-radius: 999px;
  }
  .tie-btn.selected {
    border: 1.5px solid var(--warm);
    color: var(--warm);
    background: var(--warm-soft);
    font-weight: 700;
  }

  .gloss-section {
    border-top: 1px solid var(--border-soft);
    padding-top: 18px;
    margin-top: 4px;
  }
  .gloss-section .field-label { margin-bottom: 10px; }
  .gloss-options {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }
  .gloss-chip {
    background: var(--bg);
    border: 1px solid var(--border);
    color: var(--ink-soft);
    font-size: 0.88rem;
    padding: 8px 14px;
    border-radius: 999px;
  }
  .gloss-chip.selected {
    background: var(--accent);
    border-color: var(--accent);
    color: var(--accent-ink);
    font-weight: 700;
  }

  .note-field { margin-top: 16px; }
  textarea {
    width: 100%;
    font-family: var(--sans);
    font-size: 0.92rem;
    padding: 10px 12px;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--bg);
    color: var(--ink);
    resize: vertical;
    min-height: 44px;
  }

  .nav-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 22px;
  }

  /* ---- summary / done screen ---- */
  .summary { padding: 44px 36px; text-align: center; }
  .summary .eyebrow {
    text-transform: uppercase;
    letter-spacing: 0.09em;
    font-size: 0.72rem;
    color: var(--accent);
    font-weight: 700;
    margin: 0 0 10px;
  }
  .summary h1 {
    font-family: var(--serif);
    font-size: clamp(1.6rem, 4vw, 2rem);
    margin: 0 0 14px;
  }
  .summary p { color: var(--ink-soft); max-width: 46ch; margin: 0 auto 22px; }
  .summary .stat {
    font-family: var(--serif);
    font-size: 2.6rem;
    font-weight: 600;
    color: var(--accent);
    font-variant-numeric: tabular-nums;
    line-height: 1;
    margin-bottom: 4px;
  }
  .summary .stat-label { font-size: 0.78rem; color: var(--ink-faint); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 26px; }

  .hint {
    font-size: 0.8rem;
    color: var(--ink-faint);
    margin-top: 10px;
  }

  [hidden] { display: none !important; }
</style>

<div class="shell">
  <header class="top">
    <div class="name">PidginEdu Rating Study</div>
    <div class="rater" id="raterBadge"></div>
  </header>

  <!-- SCREEN 1: intro + name -->
  <section id="screenIntro" class="card intro">
    <p class="eyebrow">Blind comparison &middot; ~7&ndash;10 minutes</p>
    <h1>Which Pidgin translation reads better to you?</h1>
    <p>
      You'll see <strong>__N__ short English sentences</strong> from Nigerian secondary-school
      material, each with two Pidgin translations, labeled only <strong>A</strong> and
      <strong>B</strong>. You won't be told which system made which -- that's deliberate,
      so your judgment isn't influenced by knowing.
    </p>
    <ul>
      <li>Pick whichever reads more natural, fluent Pidgin to you -- or mark it a tie.</li>
      <li>Some sentences include a term followed by an explanation in brackets, like
        <em>"acceptable (dem worthy of acceptance)"</em>. When that's expected, you'll also be
        asked whether each option's explanation is accurate.</li>
      <li>There's no wrong answer -- go with your own ear for the language.</li>
      <li>Your progress saves in this browser as you go, so you can close the tab and come back.</li>
    </ul>
    <p style="margin-top:22px;">
      <label class="field-label" for="raterName">Your name (so ratings from different people can be told apart)</label>
      <input type="text" id="raterName" placeholder="e.g. Amaka O." autocomplete="off">
    </p>
    <p style="margin-top:20px;">
      <button class="btn-primary" id="startBtn">Start rating</button>
    </p>
  </section>

  <!-- SCREEN 2: rating flow -->
  <section id="screenRate" hidden>
    <div class="progress-wrap">
      <div class="progress-meta">
        <span id="progressText">Item 1 of __N__</span>
        <span id="progressPct">0%</span>
      </div>
      <div class="progress-track"><div class="progress-fill" id="progressFill" style="width:0%"></div></div>
    </div>

    <div class="card item-card">
      <span class="subject-tag" id="subjectTag"></span>

      <div class="context-block">
        <div class="context-row">
          <div class="k">Source (English)</div>
          <div class="v" id="sourceText"></div>
        </div>
        <div class="context-row reference">
          <div class="k">Reference Pidgin (for context only -- not written by either model)</div>
          <div class="v" id="referenceText"></div>
        </div>
      </div>

      <label class="field-label">Which reads better?</label>
      <div class="options">
        <button class="option" id="optA" data-choice="a">
          <span class="opt-label">Option A <span class="opt-check">&#10003; better</span></span>
          <span class="opt-text" id="textA"></span>
        </button>
        <button class="option" id="optB" data-choice="b">
          <span class="opt-label">Option B <span class="opt-check">&#10003; better</span></span>
          <span class="opt-text" id="textB"></span>
        </button>
      </div>
      <div class="identical-note" id="identicalNote" hidden>
        Both options are worded identically for this sentence -- that happens
        when the two systems agree. "Tie" is the natural answer here.
      </div>
      <p class="hint" id="diffHint" style="text-align:center; margin: -10px 0 14px;">
        Highlighted words show where A and B differ -- useful when the
        difference is small.
      </p>

      <div class="tie-row">
        <button class="tie-btn" id="tieBtn">Tie -- about equally good</button>
      </div>

      <div class="gloss-section" id="glossSection" hidden>
        <label class="field-label">Is the bracketed explanation accurate and natural?</label>
        <div class="gloss-options" id="glossOptions">
          <button class="gloss-chip" data-gloss="both_good">Both good</button>
          <button class="gloss-chip" data-gloss="a_only">Only A's is good</button>
          <button class="gloss-chip" data-gloss="b_only">Only B's is good</button>
          <button class="gloss-chip" data-gloss="neither">Neither is good</button>
          <button class="gloss-chip" data-gloss="na">Not applicable</button>
        </div>
      </div>

      <div class="note-field">
        <label class="field-label" for="noteText">Anything worth flagging? (optional)</label>
        <textarea id="noteText" rows="1" placeholder="e.g. both feel unnatural, reference itself looks off, ..."></textarea>
      </div>
    </div>

    <div class="nav-row">
      <button class="btn-ghost" id="backBtn">&larr; Back</button>
      <button class="btn-primary" id="nextBtn" disabled>Next &rarr;</button>
    </div>
  </section>

  <!-- SCREEN 3: done -->
  <section id="screenDone" hidden class="card summary">
    <p class="eyebrow">All done</p>
    <div class="stat" id="doneCount">__N__</div>
    <div class="stat-label">sentences rated</div>
    <h1>Thank you<span id="doneNameSuffix"></span>.</h1>
    <p>
      Copy your results below, then paste them into a message (email, WhatsApp, however you
      got this link) and send it back to whoever asked you to do this. Nothing is sent
      automatically -- it only leaves this page when you paste it somewhere yourself.
    </p>
    <button class="btn-primary" id="copyBtn">Copy my results</button>
    <p class="hint" id="copyHint"></p>
    <textarea id="exportText" readonly style="width:100%; min-height:110px; margin-top:16px; font-family:monospace; font-size:0.78rem;"></textarea>
    <p class="hint" style="margin-top:22px;">
      <button class="btn-ghost" id="reviewBtn">Review / change my answers</button>
    </p>
  </section>
</div>

<script type="application/json" id="ratingData">__DATA_JSON__</script>

<script>
(function () {
  "use strict";

  var ROWS = JSON.parse(document.getElementById("ratingData").textContent);
  var STORAGE_KEY = "pidginedu_human_eval_v1";

  var state = {
    raterName: "",
    index: 0,
    answers: {} // row id -> {choice, gloss, note}
  };

  function loadState() {
    try {
      var raw = localStorage.getItem(STORAGE_KEY);
      if (raw) {
        var parsed = JSON.parse(raw);
        if (parsed && typeof parsed === "object") {
          state = Object.assign(state, parsed);
        }
      }
    } catch (e) { /* private mode or blocked storage: start fresh, silently */ }
  }

  function saveState() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch (e) { /* storage unavailable: in-memory only for this session */ }
  }

  var screenIntro = document.getElementById("screenIntro");
  var screenRate = document.getElementById("screenRate");
  var screenDone = document.getElementById("screenDone");
  var raterNameInput = document.getElementById("raterName");
  var startBtn = document.getElementById("startBtn");
  var raterBadge = document.getElementById("raterBadge");

  var progressText = document.getElementById("progressText");
  var progressPct = document.getElementById("progressPct");
  var progressFill = document.getElementById("progressFill");
  var subjectTag = document.getElementById("subjectTag");
  var sourceText = document.getElementById("sourceText");
  var referenceText = document.getElementById("referenceText");
  var optA = document.getElementById("optA");
  var optB = document.getElementById("optB");
  var textA = document.getElementById("textA");
  var textB = document.getElementById("textB");
  var tieBtn = document.getElementById("tieBtn");
  var glossSection = document.getElementById("glossSection");
  var glossOptions = document.getElementById("glossOptions");
  var noteText = document.getElementById("noteText");
  var backBtn = document.getElementById("backBtn");
  var nextBtn = document.getElementById("nextBtn");

  var identicalNote = document.getElementById("identicalNote");
  var diffHint = document.getElementById("diffHint");

  var doneCount = document.getElementById("doneCount");
  var doneNameSuffix = document.getElementById("doneNameSuffix");
  var copyBtn = document.getElementById("copyBtn");
  var copyHint = document.getElementById("copyHint");
  var exportText = document.getElementById("exportText");
  var reviewBtn = document.getElementById("reviewBtn");

  function escapeHtml(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  // Word-level LCS diff: highlights the words that differ between two
  // options so a rater doesn't have to spot a one-character difference
  // (e.g. "young(-20)" vs "young(0-20)") by eye, and doesn't mistake a
  // genuinely identical pair for one they just haven't compared closely.
  function wordDiff(a, b) {
    var aw = a.split(/(\s+)/);
    var bw = b.split(/(\s+)/);
    var n = aw.length, m = bw.length;
    var dp = [];
    for (var i = 0; i <= n; i++) dp.push(new Array(m + 1).fill(0));
    for (i = n - 1; i >= 0; i--) {
      for (var j = m - 1; j >= 0; j--) {
        dp[i][j] = aw[i] === bw[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
      }
    }
    var i2 = 0, j2 = 0, aOut = [], bOut = [];
    while (i2 < n && j2 < m) {
      if (aw[i2] === bw[j2]) { aOut.push([aw[i2], false]); bOut.push([bw[j2], false]); i2++; j2++; }
      else if (dp[i2 + 1][j2] >= dp[i2][j2 + 1]) { aOut.push([aw[i2], true]); i2++; }
      else { bOut.push([bw[j2], true]); j2++; }
    }
    while (i2 < n) { aOut.push([aw[i2], true]); i2++; }
    while (j2 < m) { bOut.push([bw[j2], true]); j2++; }
    return { a: aOut, b: bOut };
  }

  function renderDiffHtml(tokens) {
    var html = "";
    for (var k = 0; k < tokens.length; k++) {
      var text = tokens[k][0], changed = tokens[k][1];
      if (!text) continue;
      var isWhitespace = /^\s+$/.test(text);
      html += (changed && !isWhitespace) ? "<mark class=\"diffw\">" + escapeHtml(text) + "</mark>" : escapeHtml(text);
    }
    return html;
  }

  function currentAnswer() {
    var row = ROWS[state.index];
    return state.answers[row.id] || { choice: null, gloss: null, note: "" };
  }

  function setAnswer(patch) {
    var row = ROWS[state.index];
    var existing = currentAnswer();
    state.answers[row.id] = Object.assign({}, existing, patch);
    saveState();
    renderControls();
  }

  function renderItem() {
    var row = ROWS[state.index];
    subjectTag.textContent = row.subject;
    sourceText.textContent = row.source;
    referenceText.textContent = row.reference;
    glossSection.hidden = !row.has_gloss;

    if (row.a === row.b) {
      textA.textContent = row.a;
      textB.textContent = row.b;
      identicalNote.hidden = false;
      diffHint.hidden = true;
    } else {
      var diff = wordDiff(row.a, row.b);
      textA.innerHTML = renderDiffHtml(diff.a);
      textB.innerHTML = renderDiffHtml(diff.b);
      identicalNote.hidden = true;
      diffHint.hidden = false;
    }

    var ans = currentAnswer();
    noteText.value = ans.note || "";

    renderControls();
    updateProgress();
    window.scrollTo({ top: 0, behavior: "instant" in window ? "instant" : "auto" });
  }

  function renderControls() {
    var ans = currentAnswer();
    optA.classList.toggle("selected", ans.choice === "a");
    optB.classList.toggle("selected", ans.choice === "b");
    tieBtn.classList.toggle("selected", ans.choice === "tie");

    var chips = glossOptions.querySelectorAll(".gloss-chip");
    for (var i = 0; i < chips.length; i++) {
      chips[i].classList.toggle("selected", chips[i].dataset.gloss === ans.gloss);
    }

    var row = ROWS[state.index];
    var needsGloss = row.has_gloss;
    var ready = !!ans.choice && (!needsGloss || !!ans.gloss);
    nextBtn.disabled = !ready;
    backBtn.style.visibility = state.index === 0 ? "hidden" : "visible";
  }

  function updateProgress() {
    var n = ROWS.length;
    var pct = Math.round((state.index / n) * 100);
    progressText.textContent = "Item " + (state.index + 1) + " of " + n;
    progressPct.textContent = pct + "%";
    progressFill.style.width = pct + "%";
  }

  function showScreen(name) {
    screenIntro.hidden = name !== "intro";
    screenRate.hidden = name !== "rate";
    screenDone.hidden = name !== "done";
  }

  startBtn.addEventListener("click", function () {
    var name = raterNameInput.value.trim();
    if (!name) {
      raterNameInput.focus();
      raterNameInput.style.borderColor = "var(--danger)";
      return;
    }
    state.raterName = name;
    raterBadge.textContent = name;
    saveState();
    showScreen("rate");
    renderItem();
  });

  optA.addEventListener("click", function () { setAnswer({ choice: "a" }); });
  optB.addEventListener("click", function () { setAnswer({ choice: "b" }); });
  tieBtn.addEventListener("click", function () { setAnswer({ choice: "tie" }); });

  glossOptions.addEventListener("click", function (e) {
    var btn = e.target.closest(".gloss-chip");
    if (!btn) return;
    setAnswer({ gloss: btn.dataset.gloss });
  });

  noteText.addEventListener("input", function () {
    setAnswer({ note: noteText.value });
  });

  backBtn.addEventListener("click", function () {
    if (state.index === 0) return;
    state.index -= 1;
    saveState();
    renderItem();
  });

  nextBtn.addEventListener("click", function () {
    if (state.index < ROWS.length - 1) {
      state.index += 1;
      saveState();
      renderItem();
    } else {
      finish();
    }
  });

  function finish() {
    var answered = Object.keys(state.answers).length;
    doneCount.textContent = answered;
    doneNameSuffix.textContent = state.raterName ? ", " + state.raterName : "";
    exportText.value = JSON.stringify(buildExport(), null, 2);
    showScreen("done");
  }

  reviewBtn.addEventListener("click", function () {
    state.index = 0;
    saveState();
    showScreen("rate");
    renderItem();
  });

  function buildExport() {
    var rows = [];
    for (var i = 0; i < ROWS.length; i++) {
      var row = ROWS[i];
      var ans = state.answers[row.id] || {};
      rows.push({
        row_id: row.id,
        source_id: row.source_id,
        subject: row.subject,
        choice: ans.choice || null,
        gloss_judgment: ans.gloss || null,
        note: ans.note || ""
      });
    }
    return {
      study: "pidginedu_flagship_pairwise_v1",
      rater_name: state.raterName,
      exported_at: new Date().toISOString(),
      total_items: ROWS.length,
      answered_items: rows.filter(function (r) { return r.choice; }).length,
      ratings: rows
    };
  }

  copyBtn.addEventListener("click", async function () {
    var text = exportText.value;
    var copied = false;
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(text);
        copied = true;
      }
    } catch (e) { copied = false; }

    if (!copied) {
      // Fallback for browsers without (or blocking) the async clipboard API.
      exportText.focus();
      exportText.select();
      try { copied = document.execCommand("copy"); } catch (e) { copied = false; }
    }

    copyHint.textContent = copied
      ? "Copied! Paste it into your message now."
      : "Could not copy automatically -- tap the box below, select all, and copy manually.";
  });

  document.addEventListener("keydown", function (e) {
    if (screenRate.hidden) return;
    if (e.key === "a" || e.key === "A") optA.click();
    if (e.key === "b" || e.key === "B") optB.click();
    if (e.key === "t" || e.key === "T") tieBtn.click();
    if (e.key === "Enter" && !nextBtn.disabled) nextBtn.click();
  });

  loadState();
  if (state.raterName) {
    raterNameInput.value = state.raterName;
    raterBadge.textContent = state.raterName;
  }
  if (state.raterName && Object.keys(state.answers).length > 0 && state.index < ROWS.length) {
    showScreen("rate");
    renderItem();
  } else if (state.raterName && state.index >= ROWS.length) {
    finish();
  }
})();
</script>
'''

html = TEMPLATE.replace('__DATA_JSON__', data_json).replace('__N__', str(len(rows)))

with open('pidginedu_rating_study.html', 'w', encoding='utf-8') as f:
    f.write(html)

print('wrote', len(html), 'bytes')
