(function () {
  const picker = document.getElementById("admin-llm-picker");
  if (!picker) return;

  const trigger = document.getElementById("llm-picker-trigger");
  const menu = document.getElementById("llm-picker-menu");
  const hidden = document.getElementById("llm_model_key_input");
  const labelEl = document.getElementById("llm-picker-label");
  const subEl = document.getElementById("llm-picker-sub");
  const iconEl = document.getElementById("llm-picker-icon");
  const badge = document.getElementById("llm-active-badge");
  const hint = document.getElementById("llm-mode-hint");
  const autoHint = hint ? hint.getAttribute("data-auto-hint") || "" : "";

  function closeMenu() {
    menu.hidden = true;
    trigger.setAttribute("aria-expanded", "false");
  }

  function openMenu() {
    menu.hidden = false;
    trigger.setAttribute("aria-expanded", "true");
  }

  function setActiveOption(btn) {
    picker.querySelectorAll(".admin-llm-picker__option").forEach((el) => {
      el.classList.remove("admin-llm-picker__option--active");
      el.setAttribute("aria-selected", "false");
      const check = el.querySelector(".admin-llm-picker__check");
      if (check) check.remove();
    });
    btn.classList.add("admin-llm-picker__option--active");
    btn.setAttribute("aria-selected", "true");
    const check = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    check.setAttribute("class", "admin-llm-picker__check");
    check.setAttribute("width", "18");
    check.setAttribute("height", "18");
    check.setAttribute("viewBox", "0 0 24 24");
    check.setAttribute("fill", "none");
    check.innerHTML =
      '<path d="M5 12l5 5L19 7" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>';
    btn.appendChild(check);
  }

  function updateHint(key, label) {
    if (!hint) return;
    if (key === "auto" && autoHint) {
      hint.innerHTML =
        "<strong>Auto</strong> uses fallback: " +
        autoHint +
        ". If one provider fails, the next is tried automatically.";
    } else {
      hint.innerHTML =
        "<strong>Fixed model</strong> — every customer query uses <em>" +
        label +
        "</em> only (routing, catalog AI, answers).";
    }
  }

  trigger.addEventListener("click", function (e) {
    e.preventDefault();
    if (menu.hidden) openMenu();
    else closeMenu();
  });

  document.addEventListener("click", function (e) {
    if (!picker.contains(e.target)) closeMenu();
  });

  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") closeMenu();
  });

  picker.querySelectorAll(".admin-llm-picker__option").forEach(function (btn) {
    btn.addEventListener("click", function () {
      if (btn.disabled || btn.dataset.available === "0") return;
      const key = btn.dataset.key || "auto";
      const label = btn.dataset.label || "Auto";
      const sub = btn.dataset.sub || "";
      hidden.value = key;
      labelEl.textContent = label;
      subEl.textContent = sub;
      iconEl.textContent = key === "auto" ? "∞" : "◆";
      if (badge) {
        badge.textContent = key === "auto" ? "Auto" : "Fixed";
        badge.classList.toggle("admin-llm-badge--auto", key === "auto");
      }
      setActiveOption(btn);
      updateHint(key, label);
      closeMenu();
    });
  });

})();
