---
layout: post
title: <% tp.date.now("YYYY-MM-DD") %>-
description: ""
sitemap: true
hide_last_modified: false
image: 
related_posts: 
accent_image: 
accent_color: 
theme_color: 
invert_sidebar: false
category: "[empty, also, ok]"
---

<%* 
// JavaScript 코드를 실행하여 span 태그를 수정하는 함수
const hideSpanTags = () => {
    let spans = document.querySelectorAll('span[style*="color"]');
    spans.forEach(span => {
        span.style.color = span.getAttribute('style').match(/color:\s*([^;]+)/)[1];
        span.removeAttribute('style');
    });
};

// DOMContentLoaded 이벤트 리스너를 추가하여 페이지가 로드된 후 실행
document.addEventListener("DOMContentLoaded", hideSpanTags);
-%>
