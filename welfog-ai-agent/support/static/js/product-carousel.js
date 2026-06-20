/**
 * Product rail carousel — hide scrollbar, add prev/next arrows, keep touch scroll.
 * Enhances existing .wf-product-rail markup from chat responses (no backend changes).
 */
(function () {
    "use strict";

    var SCROLL_EDGE = 8;

    function scrollStep(rail) {
        var card = rail.querySelector(".wf-product-card");
        if (!card) {
            return Math.max(160, rail.clientWidth * 0.75);
        }
        var gap = 12;
        var cardWidth = card.getBoundingClientRect().width || 160;
        var perStep = window.matchMedia("(max-width: 767px)").matches ? 2 : 3;
        return perStep * (cardWidth + gap);
    }

    function updateCarouselArrows(carousel) {
        if (!carousel) return;
        var rail = carousel.querySelector(".wf-product-rail");
        var prev = carousel.querySelector(".wf-product-carousel__btn--prev");
        var next = carousel.querySelector(".wf-product-carousel__btn--next");
        if (!rail || !prev || !next) return;

        var maxScroll = rail.scrollWidth - rail.clientWidth;
        var atStart = rail.scrollLeft <= SCROLL_EDGE;
        var atEnd = maxScroll <= SCROLL_EDGE || rail.scrollLeft >= maxScroll - SCROLL_EDGE;
        var overflow = maxScroll > SCROLL_EDGE;

        prev.classList.toggle("is-hidden", !overflow || atStart);
        prev.disabled = !overflow || atStart;
        next.classList.toggle("is-hidden", !overflow || atEnd);
        next.disabled = !overflow || atEnd;
    }

    function bindCarousel(carousel) {
        if (!carousel || carousel.dataset.carouselBound === "1") return;
        carousel.dataset.carouselBound = "1";

        var rail = carousel.querySelector(".wf-product-rail");
        var prev = carousel.querySelector(".wf-product-carousel__btn--prev");
        var next = carousel.querySelector(".wf-product-carousel__btn--next");
        if (!rail || !prev || !next) return;

        prev.addEventListener("click", function (ev) {
            ev.preventDefault();
            ev.stopPropagation();
            rail.scrollBy({ left: -scrollStep(rail), behavior: "smooth" });
        });
        next.addEventListener("click", function (ev) {
            ev.preventDefault();
            ev.stopPropagation();
            rail.scrollBy({ left: scrollStep(rail), behavior: "smooth" });
        });
        rail.addEventListener("scroll", function () {
            updateCarouselArrows(carousel);
        }, { passive: true });

        if (typeof ResizeObserver !== "undefined") {
            var ro = new ResizeObserver(function () {
                updateCarouselArrows(carousel);
            });
            ro.observe(rail);
        }

        window.addEventListener("resize", function onResize() {
            updateCarouselArrows(carousel);
        });

        updateCarouselArrows(carousel);
        requestAnimationFrame(function () {
            updateCarouselArrows(carousel);
        });
    }

    function enhanceProductRail(rail) {
        if (!rail || rail.dataset.carouselEnhanced === "1") return null;
        if (rail.closest(".wf-product-carousel")) return rail.closest(".wf-product-carousel");

        rail.dataset.carouselEnhanced = "1";

        var carousel = document.createElement("div");
        carousel.className = "wf-product-carousel";
        rail.parentNode.insertBefore(carousel, rail);
        carousel.appendChild(rail);

        var prev = document.createElement("button");
        prev.type = "button";
        prev.className = "wf-product-carousel__btn wf-product-carousel__btn--prev is-hidden";
        prev.setAttribute("aria-label", "Previous products");
        prev.innerHTML = '<i class="fas fa-chevron-left" aria-hidden="true"></i>';

        var next = document.createElement("button");
        next.type = "button";
        next.className = "wf-product-carousel__btn wf-product-carousel__btn--next is-hidden";
        next.setAttribute("aria-label", "Next products");
        next.innerHTML = '<i class="fas fa-chevron-right" aria-hidden="true"></i>';

        carousel.insertBefore(prev, rail);
        carousel.appendChild(next);

        bindCarousel(carousel);
        return carousel;
    }

    function initProductCarousels(scope) {
        var root = scope && scope.querySelectorAll
            ? scope
            : document;
        var rails = root.querySelectorAll
            ? root.querySelectorAll(".wf-product-rail:not([data-carousel-enhanced])")
            : [];
        rails.forEach(function (rail) {
            enhanceProductRail(rail);
        });
        if (root.querySelectorAll) {
            root.querySelectorAll(".wf-product-carousel").forEach(function (carousel) {
                updateCarouselArrows(carousel);
            });
        }
    }

    window.welfogInitProductCarousels = initProductCarousels;

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", function () {
            initProductCarousels(document.getElementById("chat") || document);
        });
    } else {
        initProductCarousels(document.getElementById("chat") || document);
    }
})();
