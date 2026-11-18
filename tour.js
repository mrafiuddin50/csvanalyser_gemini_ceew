/*
 * Guided tour for the Tabular Data Analyser.
 *
 * Uses driver.js (MIT licence, loaded in index.html) to scroll to each part of the
 * page and highlight it. Sections are found by the CSS classes Streamlit adds to
 * keyed containers: st.container(key="tour-x") becomes class "st-key-tour-x".
 * Those keys live in stlite_app.py; tests/test_tour_anchors.py checks both sides match.
 *
 * Started by any element with the class "tda-tour-start" (the "Take the tour" buttons).
 */
(function () {
  "use strict";

  var DATA_TAB = "① Data & quality";
  var ASK_TAB = "② Ask questions";
  var EXPORT_TAB = "③ Export";

  var STEPS = [
    {
      key: "tour-header",
      title: "Welcome to the Tabular Data Analyser",
      text: "Ask questions about a CSV or Excel file in plain English. Python runs <b>inside your browser</b>, " +
        "so your file never leaves your computer. This tour points out each part of the page; use " +
        "<b>Next</b> and <b>Back</b>, or the arrow keys. Press Esc to leave at any time.",
    },
    {
      key: "tour-connect", sidebar: true, side: "right",
      title: "1 · Connect to Gemini",
      text: "Paste your Gemini API key (free from Google AI Studio) and press Enter. The app checks it and lists " +
        "the models your key can use. With a <b>free-tier</b> key, Google may use what is sent to improve its " +
        "products; see <i>Help</i>.",
    },
    {
      key: "tour-data", sidebar: true, side: "right", offerSample: true,
      title: "2 · Load your data",
      text: "Upload a CSV, TSV, TXT or Excel file; the app works out the encoding and separator. No file to " +
        "hand? <b>Try the synthetic sample</b>: made-up air-quality data with a few problems built in to " +
        "practise on.",
    },
    {
      key: "tour-status",
      title: "Status line",
      text: "Shows the last thing the app did, for example loading a file or answering a question, and how long " +
        "it took. While something is running, a progress box shows each step.",
    },
    {
      key: "tour-banner", needsData: true,
      title: "Data quality verdict",
      text: "As soon as data loads, the app audits it in your browser and gives two verdicts: <b>data " +
        "quality</b> (Good / Needs Attention / Poor) and <b>analysis readiness</b> (Ready / Ready with caution / " +
        "Not ready).",
    },
    {
      key: "tour-capabilities", needsData: true, tab: DATA_TAB, subTab: "Data quality audit",
      title: "What you can analyse",
      text: "For each common kind of analysis (counts, averages, comparisons, trends, correlations, joining, " +
        "maps, sharing) the audit says whether it is ready, needs care or is not possible, and why. Scroll down " +
        "in this tab for the detailed findings and a downloadable report.",
    },
    {
      key: "tour-examples", needsData: true, tab: ASK_TAB,
      title: "Example questions",
      text: "Not sure what to ask? Click an example. They are built from your column names and types only.",
    },
    {
      key: "tour-question", needsData: true, tab: ASK_TAB,
      title: "Ask your question",
      text: "Type a question and press <b>Analyse</b>. Open <i>What will be sent to Gemini</i> to see exactly " +
        "what leaves your browser: never your rows or values. If the code fails, Gemini is asked to fix it " +
        "(up to three attempts).",
    },
    {
      key: "tour-answers", needsData: true, optional: true, tab: ASK_TAB,
      title: "Your answers",
      text: "Each answer is a card. <b>Result</b> shows the table or chart; <b>Code</b> lets you edit and " +
        "<i>Run my version</i> locally, or <i>Explain this code</i>; <b>Sent to Gemini</b> records each request.",
    },
    {
      key: "tour-export-session", needsData: true, tab: EXPORT_TAB,
      title: "Download as a Jupyter notebook",
      text: "Save your questions and the code that answered them as a <b>Jupyter notebook (.ipynb)</b> or a " +
        "<b>Python script (.py)</b>. Open the notebook in Jupyter, VS Code or Google Colab and run it next to " +
        "your data file to reproduce every result. The files hold code only, never your data or results, so " +
        "they can be shared safely. The buttons become active after your first answer.",
    },
    {
      key: "tour-export-report", needsData: true, tab: EXPORT_TAB,
      title: "Data quality report",
      text: "Download the data quality audit as a report to share with colleagues. Open it in any browser; " +
        "use Print → Save as PDF if you need a PDF.",
    },
    {
      key: "tour-settings", sidebar: true, side: "right",
      title: "3 · Settings",
      text: "Turn on <b>Anonymise column names</b> if the names themselves are sensitive, edit the instructions " +
        "sent with every question, or clear your question history.",
    },
    {
      key: "tour-activity", sidebar: true, side: "right",
      title: "Activity log",
      text: "A list of everything the app has done this session, with how long each step took.",
    },
    {
      key: "tour-welcome", noDataOnly: true,
      title: "Load data to see more",
      text: "The rest of the tour (data quality, questions and export) appears once data is loaded. Try the " +
        "synthetic sample, then take the tour again.",
    },
    {
      key: "tour-help",
      title: "Help and the printable guide",
      text: "<b>Help</b> explains how your data is handled and how to keep memory use down. The " +
        "<b>printable guide</b> covers everything in more detail, with troubleshooting and practice exercises.",
    },
  ];

  var driverObj = null;
  var active = [];
  var direction = 1;

  function wait(ms) { return new Promise(function (resolve) { setTimeout(resolve, ms); }); }

  function section(key) { return document.querySelector(".st-key-" + key); }

  function isVisible(el) {
    if (!el) return false;
    var box = el.getBoundingClientRect();
    return box.width > 0 && box.height > 0 && el.offsetParent !== null;
  }

  async function waitFor(check, timeoutMs) {
    var start = Date.now();
    while (Date.now() - start < timeoutMs) {
      var value = check();
      if (value) return value;
      await wait(150);
    }
    return null;
  }

  function hasData() { return !!section("tour-tabs"); }

  function clickTab(label) {
    var tabs = document.querySelectorAll('[role="tab"]');
    for (var i = 0; i < tabs.length; i++) {
      if (tabs[i].innerText.indexOf(label) !== -1) {
        if (tabs[i].getAttribute("aria-selected") !== "true") tabs[i].click();
        return;
      }
    }
  }

  function openSidebar() {
    var opener = document.querySelector('[data-testid="stExpandSidebarButton"]') ||
      document.querySelector('[data-testid="stSidebarCollapsedControl"] button');
    if (opener && !isVisible(section("tour-connect"))) opener.click();
  }

  function buildSteps() {
    var withData = hasData();
    return STEPS.filter(function (step) {
      if (step.needsData && !withData) return false;
      if (step.noDataOnly && withData) return false;
      if (step.optional && !section(step.key)) return false;   // e.g. no answers yet
      return true;
    });
  }

  // Gets the page ready for a step (sidebar open, right tab selected); returns its element or null.
  async function prepare(step) {
    if (step.sidebar) openSidebar();
    if (step.tab) {
      clickTab(step.tab);
      if (step.subTab) { await wait(300); clickTab(step.subTab); }
    }
    var el = await waitFor(function () {
      var candidate = section(step.key);
      return isVisible(candidate) ? candidate : null;
    }, step.optional ? 1500 : (step.tab ? 10000 : 3000));
    if (el && step.tab) await wait(400);   // let the tab finish drawing before measuring it
    return el;
  }

  // Moves to the step at `index`, skipping any whose section is not on the page.
  async function go(index) {
    while (index >= 0 && index < active.length) {
      var el = await prepare(active[index]);
      if (el) {
        if (driverObj.isActive()) driverObj.moveTo(index); else driverObj.drive(index);
        return;
      }
      index += direction;
    }
    driverObj.destroy();
  }

  function addSampleButton(popover) {
    var button = document.createElement("button");
    button.type = "button";
    button.className = "tda-tour-sample";
    button.innerText = "Load the sample and continue";
    button.addEventListener("click", async function () {
      button.disabled = true;
      button.innerText = "Loading the sample…";
      var sampleButton = document.querySelector(".st-key-sample_sidebar button") ||
        document.querySelector(".st-key-sample_welcome button");
      if (!sampleButton) { button.innerText = "Sample button not found"; return; }
      sampleButton.click();
      var loaded = await waitFor(function () { return hasData() && isVisible(section("tour-banner")); }, 60000);
      driverObj.destroy();
      if (!loaded) return;
      await wait(600);
      start("tour-status");
    });
    popover.footerButtons.insertBefore(button, popover.footerButtons.firstChild);
  }

  function makeDriver() {
    return window.driver.js.driver({
      steps: active.map(function (step) {
        return {
          element: function () { return section(step.key); },
          popover: { title: step.title, description: step.text, side: step.side || "bottom", align: "start" },
        };
      }),
      showProgress: true,
      progressText: "{{current}} of {{total}}",
      nextBtnText: "Next",
      prevBtnText: "Back",
      doneBtnText: "Finish",
      popoverClass: "tda-tour",
      overlayOpacity: 0.55,
      stagePadding: 8,
      smoothScroll: true,
      allowClose: true,
      onNextClick: function () {
        var index = driverObj.getActiveIndex();
        if (index >= active.length - 1) { driverObj.destroy(); return; }
        direction = 1;
        go(index + 1);
      },
      onPrevClick: function () {
        direction = -1;
        go(driverObj.getActiveIndex() - 1);
      },
      onPopoverRender: function (popover) {
        var step = active[driverObj.getActiveIndex()];
        if (step && step.offerSample && !hasData()) addSampleButton(popover);
      },
    });
  }

  function start(fromKey) {
    if (!window.driver || !window.driver.js) {
      window.alert("The tour could not load. Please check your internet connection and reload the page.");
      return;
    }
    if (driverObj && driverObj.isActive()) driverObj.destroy();
    active = buildSteps();
    driverObj = makeDriver();
    direction = 1;
    var index = 0;
    if (fromKey) {
      for (var i = 0; i < active.length; i++) { if (active[i].key === fromKey) { index = i; break; } }
    }
    go(index);
  }

  document.addEventListener("click", function (event) {
    if (event.target.closest && event.target.closest(".tda-tour-start")) {
      event.preventDefault();
      start();
    }
  });

  window.tdaStartTour = start;   // handy for testing from the browser console
})();
