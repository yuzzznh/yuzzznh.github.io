document.addEventListener("DOMContentLoaded", function() {
    let spans = document.querySelectorAll('span[style*="color"]');
    spans.forEach(span => {
        span.style.color = span.getAttribute('style').match(/color:\s*([^;]+)/)[1];
        span.removeAttribute('style');
    });
});
