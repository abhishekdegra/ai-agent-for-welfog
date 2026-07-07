/**
 * Voice dictation for Welfog chat:
 * 1) Live typing via Web Speech API (Chrome/Edge → Google cloud)
 * 2) Fallback: record + Groq Whisper on our server (when browser reports "network" etc.)
 */
(function () {
    "use strict";

    var SPEECH_LANG = "en-IN";
    var STORAGE_KEY = "welfog_voice_mode";
    var MAX_NETWORK_RETRIES = 2;
    var RESTART_DELAY_MS = 400;

    function getRecognitionCtor() {
        return window.SpeechRecognition || window.webkitSpeechRecognition;
    }

    function normalizeTranscriptPiece(text) {
        return (text || "").replace(/\s+/g, " ");
    }

    function joinBaseAndSpoken(base, spoken) {
        var b = (base || "").trimEnd();
        var s = (spoken || "").trim();
        if (!s) return b;
        if (!b) return s;
        var needSpace = !/[\s]$/.test(base) && !/^[\s.,!?;:]/.test(s);
        return b + (needSpace ? " " : "") + s;
    }

    function initWelfogVoiceDictation(options) {
        options = options || {};
        var input = document.getElementById(options.inputId || "chatInput");
        var micBtn = document.getElementById(options.micBtnId || "chatMicBtn");
        var composerInner = document.querySelector(".composer__inner");
        var isLocked = typeof options.isLocked === "function" ? options.isLocked : function () { return false; };
        var getTranscribeUrl = typeof options.transcribeUrl === "function"
            ? options.transcribeUrl
            : function () { return "/api/voice/transcribe"; };

        if (!input || !micBtn) return null;

        var statusEl = document.createElement("div");
        statusEl.className = "composer-voice-status";
        statusEl.setAttribute("role", "status");
        statusEl.setAttribute("aria-live", "polite");
        statusEl.hidden = true;
        if (composerInner && composerInner.parentNode) {
            composerInner.parentNode.insertBefore(statusEl, composerInner.nextSibling);
        }

        function showStatus(msg, isError) {
            if (!msg) {
                statusEl.hidden = true;
                statusEl.textContent = "";
                statusEl.classList.remove("composer-voice-status--error");
                return;
            }
            statusEl.hidden = false;
            statusEl.textContent = msg;
            statusEl.classList.toggle("composer-voice-status--error", !!isError);
        }

        var Recognition = getRecognitionCtor();
        var browserSttSupported = !!Recognition;
        var voiceMode = localStorage.getItem(STORAGE_KEY) || "auto";

        var recognition = null;
        if (browserSttSupported) {
            recognition = new Recognition();
            recognition.continuous = false;
            recognition.interimResults = true;
            recognition.maxAlternatives = 1;
            recognition.lang = SPEECH_LANG;
        }

        var listening = false;
        var manualStop = false;
        var dictationBase = "";
        var finalTranscript = "";
        var disabledExternal = false;
        var networkRetryCount = 0;
        var restartTimer = null;
        var useServerMode = voiceMode === "server";

        var mediaRecorder = null;
        var mediaStream = null;
        var recordedChunks = [];
        var serverRecording = false;

        function setListeningState(on) {
            listening = on;
            micBtn.classList.toggle("composer__mic--active", on);
            micBtn.setAttribute("aria-pressed", on ? "true" : "false");
            if (composerInner) {
                composerInner.classList.toggle("composer__inner--dictating", on);
            }
            if (on && serverRecording) {
                micBtn.title = "Stop recording and transcribe";
            } else if (on) {
                micBtn.title = "Stop dictation";
            } else {
                micBtn.title = "Dictate (Ctrl+Shift+D)";
            }
            micBtn.setAttribute("aria-label", on ? "Stop voice input" : "Start voice dictation");
        }

        function stopMediaTracks() {
            if (mediaStream) {
                mediaStream.getTracks().forEach(function (t) { t.stop(); });
                mediaStream = null;
            }
        }

        function stopDictation() {
            manualStop = true;
            if (restartTimer) {
                clearTimeout(restartTimer);
                restartTimer = null;
            }
            if (recognition && listening && !serverRecording) {
                try {
                    recognition.stop();
                } catch (e) { /* ignore */ }
            }
            if (mediaRecorder && mediaRecorder.state !== "inactive") {
                try {
                    mediaRecorder.stop();
                } catch (e2) { /* ignore */ }
            }
            if (!serverRecording) {
                setListeningState(false);
                showStatus("");
            }
        }

        function appendTranscriptToInput(text) {
            var t = (text || "").trim();
            if (!t) return;
            input.value = joinBaseAndSpoken(dictationBase, t);
            finalTranscript = t;
        }

        function chooseMimeType() {
            var types = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"];
            for (var i = 0; i < types.length; i++) {
                if (window.MediaRecorder && MediaRecorder.isTypeSupported(types[i])) {
                    return types[i];
                }
            }
            return "";
        }

        function uploadRecording(blob, mime) {
            showStatus("Transcribing…");
            var fd = new FormData();
            fd.append("audio", blob, "voice.webm");
            var url = getTranscribeUrl();
            return fetch(url, { method: "POST", body: fd })
                .then(function (res) { return res.json().then(function (body) { return { res: res, body: body }; }); })
                .then(function (pack) {
                    if (pack.body && pack.body.ok && pack.body.text) {
                        appendTranscriptToInput(pack.body.text);
                        showStatus("");
                        return true;
                    }
                    showStatus("Could not transcribe audio. Type your message or try again.", true);
                    return false;
                })
                .catch(function () {
                    showStatus("Voice server unreachable. Check internet and try again.", true);
                    return false;
                })
                .finally(function () {
                    serverRecording = false;
                    setListeningState(false);
                    stopMediaTracks();
                });
        }

        function startServerRecording() {
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                showStatus("Microphone not available in this browser.", true);
                return Promise.resolve();
            }
            return navigator.mediaDevices.getUserMedia({ audio: true })
                .then(function (stream) {
                    mediaStream = stream;
                    recordedChunks = [];
                    var mime = chooseMimeType();
                    var opts = mime ? { mimeType: mime } : undefined;
                    mediaRecorder = new MediaRecorder(stream, opts);
                    mediaRecorder.ondataavailable = function (ev) {
                        if (ev.data && ev.data.size > 0) {
                            recordedChunks.push(ev.data);
                        }
                    };
                    mediaRecorder.onstop = function () {
                        var type = (mediaRecorder && mediaRecorder.mimeType) || mime || "audio/webm";
                        var blob = new Blob(recordedChunks, { type: type });
                        if (blob.size < 200) {
                            showStatus("No speech detected. Try again.", true);
                            serverRecording = false;
                            setListeningState(false);
                            stopMediaTracks();
                            return;
                        }
                        uploadRecording(blob, type);
                    };
                    mediaRecorder.start(250);
                    serverRecording = true;
                    setListeningState(true);
                    showStatus("Recording… speak in English or Hinglish, then tap mic again.");
                })
                .catch(function (err) {
                    serverRecording = false;
                    setListeningState(false);
                    if (err && err.name === "NotAllowedError") {
                        showStatus("Microphone access denied — allow mic in browser settings.", true);
                    } else {
                        showStatus("Could not start microphone.", true);
                    }
                });
        }

        function scheduleRecognitionRestart() {
            if (!listening || manualStop || disabledExternal || isLocked() || useServerMode) {
                return;
            }
            if (restartTimer) clearTimeout(restartTimer);
            restartTimer = setTimeout(function () {
                restartTimer = null;
                if (!listening || manualStop) return;
                try {
                    recognition.start();
                } catch (e) {
                    setListeningState(false);
                }
            }, RESTART_DELAY_MS);
        }

        function switchToServerMode(reason) {
            useServerMode = true;
            localStorage.setItem(STORAGE_KEY, "server");
            if (reason) {
                showStatus(reason + " Using server voice (tap mic when done speaking).", false);
            }
            networkRetryCount = 0;
            if (recognition) {
                try { recognition.stop(); } catch (e3) { /* ignore */ }
            }
            setListeningState(false);
            listening = false;
            manualStop = false;
            dictationBase = input.value || "";
            finalTranscript = "";
            return startServerRecording();
        }

        function startBrowserDictation() {
            manualStop = false;
            networkRetryCount = 0;
            dictationBase = input.value || "";
            finalTranscript = "";
            showStatus("Listening… speak in English or Hinglish.");
            try {
                recognition.start();
                listening = true;
                setListeningState(true);
                input.focus();
            } catch (err) {
                listening = false;
                setListeningState(false);
                if (err && err.name === "InvalidStateError") {
                    return;
                }
                if (err && err.name === "NotAllowedError") {
                    showStatus("Microphone access denied — allow mic in browser settings.", true);
                } else {
                    switchToServerMode("Live voice unavailable.");
                }
            }
        }

        function startDictation() {
            if (disabledExternal || isLocked()) return;
            if (listening) {
                stopDictation();
                return;
            }

            if (useServerMode || !browserSttSupported) {
                dictationBase = input.value || "";
                finalTranscript = "";
                manualStop = false;
                return startServerRecording();
            }

            startBrowserDictation();
        }

        if (recognition) {
            recognition.onresult = function (event) {
                networkRetryCount = 0;
                var interim = "";
                for (var i = event.resultIndex; i < event.results.length; i++) {
                    var piece = normalizeTranscriptPiece(event.results[i][0].transcript);
                    if (!piece) continue;
                    if (event.results[i].isFinal) {
                        finalTranscript += piece;
                    } else {
                        interim += piece;
                    }
                }
                var spoken = normalizeTranscriptPiece(finalTranscript + interim);
                input.value = joinBaseAndSpoken(dictationBase, spoken);
            };

            recognition.onerror = function (event) {
                var code = event.error || "";
                if (code === "aborted" || code === "no-speech") {
                    return;
                }
                if (code === "not-allowed" || code === "service-not-allowed") {
                    showStatus("Microphone access denied — allow mic in browser settings.", true);
                    stopDictation();
                    return;
                }
                if (code === "network" || code === "service-not-available") {
                    if (networkRetryCount < MAX_NETWORK_RETRIES) {
                        networkRetryCount += 1;
                        showStatus("Connecting voice… retry " + networkRetryCount + "/" + MAX_NETWORK_RETRIES);
                        try { recognition.stop(); } catch (e4) { /* ignore */ }
                        scheduleRecognitionRestart();
                        return;
                    }
                    listening = false;
                    manualStop = false;
                    switchToServerMode(
                        "Browser voice could not reach Google (network/firewall)."
                    );
                    return;
                }
                showStatus("Voice error: " + code + ". Try again or type your message.", true);
                stopDictation();
            };

            recognition.onend = function () {
                if (manualStop || !listening || useServerMode) {
                    if (!serverRecording) {
                        setListeningState(false);
                        showStatus("");
                    }
                    return;
                }
                if (disabledExternal || isLocked()) {
                    setListeningState(false);
                    showStatus("");
                    return;
                }
                scheduleRecognitionRestart();
            };
        }

        if (!browserSttSupported) {
            useServerMode = true;
            micBtn.title = "Voice input (server)";
        }

        micBtn.addEventListener("click", function (ev) {
            ev.preventDefault();
            if (disabledExternal || isLocked()) return;
            startDictation();
        });

        document.addEventListener("keydown", function (ev) {
            if (!(ev.ctrlKey && ev.shiftKey && (ev.key === "D" || ev.key === "d"))) return;
            if (disabledExternal || isLocked()) return;
            var tag = (ev.target && ev.target.tagName) || "";
            if (tag === "INPUT" || tag === "TEXTAREA" || (ev.target && ev.target.isContentEditable)) {
                if (ev.target !== input && ev.target.id !== "chatInput") return;
            }
            ev.preventDefault();
            startDictation();
        });

        function setDisabled(disabled) {
            disabledExternal = !!disabled;
            micBtn.disabled = disabledExternal;
            if (disabledExternal && listening) {
                stopDictation();
            }
        }

        return {
            supported: true,
            stop: stopDictation,
            setDisabled: setDisabled,
            isListening: function () { return listening; }
        };
    }

    window.initWelfogVoiceDictation = initWelfogVoiceDictation;
})();
