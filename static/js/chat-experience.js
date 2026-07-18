/**
 * Welfog chat experience — mobile WebView UI only.
 * Does not touch APIs, AI, or response payloads.
 */
(function (global) {
  "use strict";

  var STAGE_DELAY_MS = 450;
  var NEAR_BOTTOM_PX = 96;
  var MSG_ANIM_MS = 220;

  function prefersReducedMotion() {
    try {
      return !!(
        window.matchMedia &&
        window.matchMedia("(prefers-reduced-motion: reduce)").matches
      );
    } catch (e) {
      return false;
    }
  }

  /* ------------------------------------------------------------------ */
  /* AutoScrollManager                                                   */
  /* ------------------------------------------------------------------ */
  function AutoScrollManager(scrollEl) {
    this.el = scrollEl;
    this.pinned = true;
    this._onScroll = this._onScroll.bind(this);
    if (scrollEl) {
      scrollEl.addEventListener("scroll", this._onScroll, { passive: true });
    }
  }

  AutoScrollManager.prototype._onScroll = function () {
    if (!this.el) return;
    var distance =
      this.el.scrollHeight - this.el.scrollTop - this.el.clientHeight;
    this.pinned = distance <= NEAR_BOTTOM_PX;
  };

  AutoScrollManager.prototype.isPinned = function () {
    return this.pinned;
  };

  AutoScrollManager.prototype.pin = function () {
    this.pinned = true;
  };

  AutoScrollManager.prototype.scrollToBottom = function (opts) {
    if (!this.el) return;
    var force = !!(opts && opts.force);
    if (!force && !this.pinned) return;
    this.pinned = true;
    var el = this.el;
    var smooth = !prefersReducedMotion() && !(opts && opts.instant);
    var run = function () {
      if (smooth && typeof el.scrollTo === "function") {
        try {
          el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
          return;
        } catch (e) {}
      }
      el.scrollTop = el.scrollHeight;
    };
    requestAnimationFrame(function () {
      requestAnimationFrame(run);
    });
  };

  /* ------------------------------------------------------------------ */
  /* KeyboardAwareChatView                                               */
  /* ------------------------------------------------------------------ */
  function KeyboardAwareChatView(rootEl, scrollManager) {
    this.root = rootEl || document.documentElement;
    this.scroll = scrollManager;
    this._onViewport = this._onViewport.bind(this);
    this._lastInset = 0;
  }

  KeyboardAwareChatView.prototype.start = function () {
    var vv = window.visualViewport;
    if (vv) {
      vv.addEventListener("resize", this._onViewport);
      vv.addEventListener("scroll", this._onViewport);
    }
    window.addEventListener("focusin", this._onViewport);
    window.addEventListener("resize", this._onViewport);
    this._onViewport();
  };

  KeyboardAwareChatView.prototype._onViewport = function () {
    var inset = 0;
    var vv = window.visualViewport;
    if (vv) {
      inset = Math.max(0, window.innerHeight - vv.height - vv.offsetTop);
    }
    this.root.style.setProperty("--wf-keyboard-inset", inset + "px");
    if (Math.abs(inset - this._lastInset) > 8) {
      this._lastInset = inset;
      if (this.scroll) {
        this.scroll.scrollToBottom({ force: inset > 40 });
      }
    }
  };

  /* ------------------------------------------------------------------ */
  /* TypingStageAnimator — UI labels from user text only (no API)        */
  /* ------------------------------------------------------------------ */
  function inferThinkingStage(userMessage) {
    var t = String(userMessage || "").toLowerCase();
    if (!t) return "Thinking…";

    if (
      /\b(refund|return\s+money|paise\s+wapas|paisa\s+wapas|refund\s+status)\b/.test(
        t
      ) ||
      /\b(refund|return)\b/.test(t) && /\b(status|kab|milega|hua)\b/.test(t)
    ) {
      return "Checking refund status…";
    }

    if (
      /\b(order|track|tracking|invoice|bill|receipt|delivery\s+status)\b/.test(
        t
      ) ||
      /\b(order|ordr)\b/.test(t) &&
        /\b(history|status|kahan|kab|details|dikha|bata)\b/.test(t) ||
      /\b\d{6,20}\b/.test(t) && /\b(order|track|status|invoice)\b/.test(t)
    ) {
      return "Fetching your order…";
    }

    if (
      /\b(pincode|pin\s*code|deliver(y|able)|serviceable)\b/.test(t) ||
      /\b\d{6}\b/.test(t) && /\b(pin|deliver|aa\s*jayega)\b/.test(t)
    ) {
      return "Checking delivery…";
    }

    if (
      /\b(product|products|cover|case|shirt|tshirt|shoes|sneaker|jeans|buy|dikhao|dikha|chahiye|show\s+me|under\s+\d)\b/.test(
        t
      ) ||
      /\b(search|catalog|wishlist)\b/.test(t)
    ) {
      return "Searching products…";
    }

    if (
      /\b(welfog|policy|policies|return\s+policy|shipping|payment|seller|faq|company|about|kya\s+hai|krta\s+kya|support|customer\s+care)\b/.test(
        t
      ) ||
      /\b(what\s+is|who\s+is|tell\s+me\s+about|batao|bata)\b/.test(t)
    ) {
      return "Searching knowledge…";
    }

    return "Thinking…";
  }

  function TypingStageAnimator(labelEl) {
    this.labelEl = labelEl;
    this._timer = null;
    this._userMsg = "";
  }

  TypingStageAnimator.prototype.start = function (userMessage) {
    this.stop();
    this._userMsg = userMessage || "";
    if (!this.labelEl) return;
    this.labelEl.textContent = "";
    this.labelEl.hidden = true;
    if (prefersReducedMotion()) {
      this.labelEl.textContent = inferThinkingStage(this._userMsg);
      this.labelEl.hidden = false;
      return;
    }
    var self = this;
    this._timer = setTimeout(function () {
      self._timer = null;
      if (!self.labelEl) return;
      self.labelEl.textContent = inferThinkingStage(self._userMsg);
      self.labelEl.hidden = false;
    }, STAGE_DELAY_MS);
  };

  TypingStageAnimator.prototype.stop = function () {
    if (this._timer) {
      clearTimeout(this._timer);
      this._timer = null;
    }
  };

  /* ------------------------------------------------------------------ */
  /* ChatLoadingBubble + ThinkingIndicator                               */
  /* ------------------------------------------------------------------ */
  function ChatLoadingBubble(chatEl, scrollManager) {
    this.chatEl = chatEl;
    this.scroll = scrollManager;
    this.node = null;
    this.stage = null;
  }

  ChatLoadingBubble.prototype.show = function (userMessage) {
    this.hide();
    if (!this.chatEl) return;

    var wrap = document.createElement("div");
    wrap.className = "msg bot wf-loading-bubble";
    wrap.id = "typing";
    wrap.setAttribute("role", "status");
    wrap.setAttribute("aria-live", "polite");
    wrap.setAttribute("aria-label", "Assistant is thinking");

    wrap.innerHTML =
      '<div class="wf-thinking" aria-hidden="true">' +
      '<span class="wf-thinking__orb"></span>' +
      '<span class="wf-thinking__dots">' +
      '<span class="wf-thinking__dot"></span>' +
      '<span class="wf-thinking__dot"></span>' +
      '<span class="wf-thinking__dot"></span>' +
      "</span>" +
      "</div>" +
      '<div class="wf-thinking__stage" hidden></div>';

    this.chatEl.appendChild(wrap);
    this.node = wrap;
    this.stage = new TypingStageAnimator(
      wrap.querySelector(".wf-thinking__stage")
    );
    this.stage.start(userMessage);
    if (this.scroll) {
      this.scroll.pin();
      this.scroll.scrollToBottom({ force: true });
    }
  };

  ChatLoadingBubble.prototype.hide = function () {
    if (this.stage) {
      this.stage.stop();
      this.stage = null;
    }
    if (this.node && this.node.parentNode) {
      this.node.parentNode.removeChild(this.node);
    }
    this.node = null;
    var legacy = document.getElementById("typing");
    if (legacy && legacy.parentNode) {
      legacy.parentNode.removeChild(legacy);
    }
  };

  /* ------------------------------------------------------------------ */
  /* MessageTransition                                                   */
  /* ------------------------------------------------------------------ */
  function animateMessageIn(el) {
    if (!el) return;
    if (prefersReducedMotion()) {
      el.classList.add("wf-msg--visible");
      return;
    }
    el.classList.add("wf-msg--enter");
    requestAnimationFrame(function () {
      requestAnimationFrame(function () {
        el.classList.add("wf-msg--visible");
        el.classList.remove("wf-msg--enter");
        window.setTimeout(function () {
          /* keep wf-msg--visible for layout stability */
        }, MSG_ANIM_MS + 40);
      });
    });
  }

  /* ------------------------------------------------------------------ */
  /* Public API                                                          */
  /* ------------------------------------------------------------------ */
  function createChatExperience(options) {
    options = options || {};
    var chatEl = options.chatEl || document.getElementById("chat");
    var scroll = new AutoScrollManager(chatEl);
    var keyboard = new KeyboardAwareChatView(
      document.documentElement,
      scroll
    );
    var loading = new ChatLoadingBubble(chatEl, scroll);
    keyboard.start();

    return {
      scroll: scroll,
      keyboard: keyboard,
      loading: loading,
      showThinking: function (userMessage) {
        loading.show(userMessage);
      },
      hideThinking: function () {
        loading.hide();
      },
      appendMessage: function (el, opts) {
        if (!chatEl || !el) return;
        var force = !!(opts && opts.forceScroll);
        chatEl.appendChild(el);
        if (opts && opts.animate !== false) {
          animateMessageIn(el);
        } else {
          el.classList.add("wf-msg--visible");
        }
        if (force || scroll.isPinned()) {
          scroll.pin();
          scroll.scrollToBottom({ force: true });
        }
      },
      scrollToLatest: function (force) {
        scroll.scrollToBottom({ force: !!force });
      },
    };
  }

  global.WelfogChatExperience = {
    create: createChatExperience,
    prefersReducedMotion: prefersReducedMotion,
    inferThinkingStage: inferThinkingStage,
  };
})(window);
