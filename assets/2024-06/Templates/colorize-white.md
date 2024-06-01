<%_* 
const selectedText = tp.file.selection();
if (selectedText) {
    // 모든 <span> 태그를 제거하여 원래 텍스트로 변환
    const textWithoutSpan = selectedText.replace(/<span style='color: [^']+'>(.*?)<\/span>/g, '$1');

    tR += textWithoutSpan;
} 
_%>
