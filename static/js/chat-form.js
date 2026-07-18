(function () {
    "use strict";

    const SUPPORT_API = "/api/support/contact";
    const PROFILE_CACHE_KEY = "welfog_support_profile";
    const REQUEST_TIMEOUT_MS = 12000;

    function safeJson(value) {
        try {
            return value ? JSON.parse(value) : null;
        } catch (_error) {
            return null;
        }
    }

    function normalizeProfile(candidate) {
        if (!candidate || typeof candidate !== "object") return {};
        const source = candidate.user && typeof candidate.user === "object"
            ? candidate.user
            : candidate;
        return {
            name: String(source.name || source.full_name || source.fullName || "").trim(),
            email: String(source.email || source.email_address || "").trim(),
            phone: String(source.phone || source.mobile || source.phone_number || "").trim(),
        };
    }

    function mergeProfile(base, candidate) {
        const next = normalizeProfile(candidate);
        return {
            name: base.name || next.name,
            email: base.email || next.email,
            phone: base.phone || next.phone,
        };
    }

    function getCachedProfile() {
        let profile = { name: "", email: "", phone: "" };
        profile = mergeProfile(profile, window.WELFOG_USER);
        profile = mergeProfile(profile, window.__WELFOG_USER__);

        try {
            profile = mergeProfile(profile, safeJson(sessionStorage.getItem(PROFILE_CACHE_KEY)));
            profile = mergeProfile(profile, safeJson(localStorage.getItem(PROFILE_CACHE_KEY)));
            profile = mergeProfile(profile, safeJson(localStorage.getItem("welfog_user")));
            profile = mergeProfile(profile, safeJson(localStorage.getItem("welfog_profile")));
        } catch (_error) {
            // Storage can be unavailable in privacy-restricted WebViews.
        }

        const params = new URLSearchParams(window.location.search);
        profile = mergeProfile(profile, {
            name: params.get("name"),
            email: params.get("email"),
            phone: params.get("phone"),
        });
        return profile;
    }

    function cacheProfile(values) {
        const profile = normalizeProfile(values);
        try {
            localStorage.setItem(PROFILE_CACHE_KEY, JSON.stringify(profile));
            sessionStorage.setItem(PROFILE_CACHE_KEY, JSON.stringify(profile));
        } catch (_error) {
            // Submission must still succeed when browser storage is disabled.
        }
    }

    function makeElement(tag, className, text) {
        const element = document.createElement(tag);
        if (className) element.className = className;
        if (text !== undefined) element.textContent = text;
        return element;
    }

    function emailIsValid(value) {
        return /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/i.test(value);
    }

    function phoneIsValid(value) {
        if (!value) return true;
        const digits = value.replace(/\D/g, "");
        return digits.length >= 7 && digits.length <= 15;
    }

    class ChatForm {
        constructor(options) {
            this.options = options || {};
            this.fields = Array.isArray(this.options.fields) ? this.options.fields : [];
            this.values = Object.assign({}, this.options.prefill || {});
            this.touched = new Set();
            this.controls = {};
            this.errors = {};
            this.submitting = false;
            this.succeeded = false;
            this.root = null;
            this.body = null;
            this.status = null;
            this.submitButton = null;
            this.retryButton = null;
        }

        mount(host) {
            if (!host) throw new Error("ChatForm requires a host element.");
            this.root = makeElement("section", "wf-chat-form");
            this.root.dataset.formType = this.options.type || "generic";
            this.root.setAttribute("aria-label", this.options.title || "Chat form");

            const header = makeElement("div", "wf-chat-form__header");
            const icon = makeElement("span", "wf-chat-form__icon");
            icon.setAttribute("aria-hidden", "true");
            icon.innerHTML = this.options.icon || '<i class="fas fa-headset"></i>';
            const headingWrap = makeElement("div", "wf-chat-form__heading");
            headingWrap.appendChild(makeElement("h2", "wf-chat-form__title", this.options.title || "Form"));
            if (this.options.description) {
                headingWrap.appendChild(
                    makeElement("p", "wf-chat-form__description", this.options.description)
                );
            }
            header.appendChild(icon);
            header.appendChild(headingWrap);
            this.root.appendChild(header);

            this.status = makeElement("div", "wf-chat-form__status");
            this.status.setAttribute("aria-live", "polite");
            this.root.appendChild(this.status);

            this.body = makeElement("form", "wf-chat-form__body");
            this.body.noValidate = true;
            this.fields.forEach((field) => this.body.appendChild(this.buildField(field)));

            const actions = makeElement("div", "wf-chat-form__actions");
            this.submitButton = makeElement(
                "button",
                "wf-chat-form__submit",
                this.options.submitLabel || "Submit"
            );
            this.submitButton.type = "submit";
            actions.appendChild(this.submitButton);

            this.retryButton = makeElement("button", "wf-chat-form__retry", "Retry");
            this.retryButton.type = "button";
            this.retryButton.hidden = true;
            this.retryButton.addEventListener("click", () => this.submit());
            actions.appendChild(this.retryButton);
            this.body.appendChild(actions);

            this.body.addEventListener("submit", (event) => {
                event.preventDefault();
                this.submit();
            });
            this.root.appendChild(this.body);
            host.appendChild(this.root);

            requestAnimationFrame(() => this.root.classList.add("wf-chat-form--visible"));
            return this;
        }

        buildField(field) {
            const group = makeElement("div", "wf-chat-form__field");
            const id = "wf-chat-form-" + (this.options.type || "form") + "-" + field.name +
                "-" + Math.random().toString(36).slice(2, 7);
            const label = makeElement("label", "wf-chat-form__label");
            label.htmlFor = id;
            label.appendChild(document.createTextNode(field.label));
            if (field.required) {
                const required = makeElement("span", "wf-chat-form__required", " *");
                required.setAttribute("aria-hidden", "true");
                label.appendChild(required);
            }

            const control = document.createElement(field.multiline ? "textarea" : "input");
            control.id = id;
            control.name = field.name;
            control.className = "wf-chat-form__control";
            control.placeholder = field.placeholder || "";
            control.required = !!field.required;
            control.autocomplete = field.autocomplete || "off";
            if (!field.multiline) control.type = field.type || "text";
            if (field.multiline) control.rows = field.rows || 4;
            control.value = String(this.values[field.name] || "");
            this.values[field.name] = control.value;

            const error = makeElement("div", "wf-chat-form__error");
            error.id = id + "-error";
            control.setAttribute("aria-describedby", error.id);

            control.addEventListener("input", () => {
                this.values[field.name] = control.value;
                if (this.touched.has(field.name)) this.validateField(field);
                this.clearStatus();
            });
            control.addEventListener("blur", () => {
                this.touched.add(field.name);
                this.validateField(field);
            });

            this.controls[field.name] = control;
            this.errors[field.name] = error;
            group.appendChild(label);
            group.appendChild(control);
            group.appendChild(error);
            return group;
        }

        validateField(field) {
            const control = this.controls[field.name];
            const value = String(control.value || "").trim();
            let message = "";
            if (field.required && !value) {
                message = field.label + " is required.";
            } else if (field.validate) {
                message = field.validate(value, this.values) || "";
            }
            this.errors[field.name].textContent = message;
            control.classList.toggle("wf-chat-form__control--invalid", !!message);
            control.setAttribute("aria-invalid", message ? "true" : "false");
            return !message;
        }

        validate() {
            let valid = true;
            this.fields.forEach((field) => {
                this.touched.add(field.name);
                if (!this.validateField(field)) valid = false;
            });
            if (!valid) {
                const firstInvalid = this.body.querySelector(".wf-chat-form__control--invalid");
                if (firstInvalid) firstInvalid.focus({ preventScroll: true });
            }
            return valid;
        }

        clearStatus() {
            if (this.submitting || this.succeeded) return;
            this.status.textContent = "";
            this.status.className = "wf-chat-form__status";
            this.retryButton.hidden = true;
        }

        setSubmitting(submitting) {
            this.submitting = submitting;
            this.root.classList.toggle("wf-chat-form--submitting", submitting);
            this.root.setAttribute("aria-busy", submitting ? "true" : "false");
            this.submitButton.disabled = submitting || this.succeeded;
            this.fields.forEach((field) => {
                this.controls[field.name].readOnly = submitting || this.succeeded;
            });
            if (submitting) {
                this.submitButton.innerHTML =
                    '<span class="wf-chat-form__spinner" aria-hidden="true"></span><span>Submitting…</span>';
                this.retryButton.hidden = true;
            } else {
                this.submitButton.textContent = this.options.submitLabel || "Submit";
            }
        }

        async submit() {
            if (this.submitting || this.succeeded || !this.validate()) return;
            this.setSubmitting(true);
            this.status.textContent = "";
            this.status.className = "wf-chat-form__status";
            try {
                const result = await this.options.onSubmit(Object.assign({}, this.values));
                this.showSuccess(result);
            } catch (error) {
                this.setSubmitting(false);
                this.status.className = "wf-chat-form__status wf-chat-form__status--error";
                this.status.textContent = error && error.message
                    ? error.message
                    : "We couldn't submit your request. Please try again.";
                this.retryButton.hidden = false;
            }
        }

        showSuccess(result) {
            this.succeeded = true;
            this.setSubmitting(false);
            this.root.classList.add("wf-chat-form--success", "wf-chat-form--collapsed");
            this.status.className = "wf-chat-form__status wf-chat-form__status--success";
            this.status.replaceChildren();

            const badge = makeElement("div", "wf-chat-form__success-badge");
            badge.innerHTML = '<i class="fas fa-check-circle" aria-hidden="true"></i>';
            badge.appendChild(document.createTextNode(
                (result && result.title) || "Support request submitted"
            ));
            const detail = makeElement(
                "p",
                "wf-chat-form__success-detail",
                (result && result.message) ||
                    "Ticket details have been shared with our support team. We'll contact you soon."
            );
            const toggle = makeElement("button", "wf-chat-form__success-toggle", "View submitted details");
            toggle.type = "button";
            toggle.setAttribute("aria-expanded", "false");
            toggle.addEventListener("click", () => {
                const collapsed = this.root.classList.toggle("wf-chat-form--collapsed");
                toggle.textContent = collapsed ? "View submitted details" : "Hide submitted details";
                toggle.setAttribute("aria-expanded", collapsed ? "false" : "true");
            });
            this.status.appendChild(badge);
            this.status.appendChild(detail);
            this.status.appendChild(toggle);
            if (this.options.onSuccess) this.options.onSuccess(this.values, result);
        }
    }

    async function submitSupportRequest(values) {
        const controller = new AbortController();
        const timer = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
        let response;
        try {
            response = await fetch(SUPPORT_API, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    name: String(values.name || "").trim(),
                    email: String(values.email || "").trim(),
                    phone: String(values.phone || "").trim(),
                    message: String(values.message || "").trim(),
                }),
                signal: controller.signal,
            });
        } catch (error) {
            if (error && error.name === "AbortError") {
                throw new Error("The request timed out. Your details are safe — tap Retry.");
            }
            throw new Error("Network error. Check your connection and tap Retry.");
        } finally {
            window.clearTimeout(timer);
        }

        let data;
        try {
            const raw = await response.text();
            data = raw ? JSON.parse(raw) : null;
        } catch (_error) {
            throw new Error("The support service returned an invalid response. Please retry.");
        }
        if (!response.ok || !data || typeof data !== "object" ||
            data.success === false || data.status === false) {
            const apiMessage = data && (data.message || data.error);
            throw new Error(apiMessage || "Support request could not be submitted. Please retry.");
        }
        return {
            title: "Support request submitted successfully",
            message: "Ticket details have been shared with our support team. We'll contact you soon.",
            response: data,
        };
    }

    function createSupportForm(host, options) {
        const config = options || {};
        const profile = Object.assign(getCachedProfile(), config.prefill || {});
        return new ChatForm({
            type: "support",
            title: "Contact Welfog Support",
            description: "Tell us what happened. Our support team will get back to you.",
            icon: '<i class="fas fa-headset"></i>',
            prefill: profile,
            submitLabel: "Submit request",
            fields: [
                {
                    name: "name",
                    label: "Name",
                    required: true,
                    autocomplete: "name",
                    placeholder: "Your name",
                    validate: (value) => value.length >= 2 ? "" : "Enter a valid name.",
                },
                {
                    name: "email",
                    label: "Email",
                    type: "email",
                    required: true,
                    autocomplete: "email",
                    placeholder: "you@example.com",
                    validate: (value) => emailIsValid(value) ? "" : "Enter a valid email address.",
                },
                {
                    name: "phone",
                    label: "Phone",
                    type: "tel",
                    autocomplete: "tel",
                    placeholder: "+91 98765 43210",
                    validate: (value) => phoneIsValid(value) ? "" : "Enter a valid phone number.",
                },
                {
                    name: "message",
                    label: "Message",
                    required: true,
                    multiline: true,
                    rows: 4,
                    autocomplete: "off",
                    placeholder: "How can we help?",
                    validate: (value) => value.length >= 5 ? "" : "Please add a little more detail.",
                },
            ],
            onSubmit: submitSupportRequest,
            onSuccess: (values, result) => {
                cacheProfile(values);
                if (config.onSuccess) config.onSuccess(values, result);
            },
        }).mount(host);
    }

    function shouldOpenForMessage(message) {
        const text = String(message || "").trim().toLowerCase();
        if (!text || text.length > 220) return false;
        return /^(?:(?:i\s+(?:have|want\s+to\s+(?:file|make)|need\s+to\s+(?:file|make))\s+(?:a\s+)?)|(?:mujhe\s+)|(?:main\s+))?(?:complaint|complain|report\s+(?:a\s+)?problem|report\s+(?:an\s+)?issue|raise\s+(?:a\s+)?complaint|contact\s+support|customer\s+support|support\s+request|complaint\s+karni\s+hai|problem\s+report\s+karna\s+hai)(?:\b|$)/i.test(text);
    }

    window.WelfogChatForms = {
        ChatForm: ChatForm,
        createSupportForm: createSupportForm,
        getCachedProfile: getCachedProfile,
        shouldOpenSupportForMessage: shouldOpenForMessage,
    };
})();
