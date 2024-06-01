<%_* 
const selectedText = tp.file.selection().trim();
if (selectedText) {
    const color = '#f8dd74'; // 일반 텍스트에 적용할 색상
    const innerColor = '#f8dd74'; // 마크다운 문법이 적용된 텍스트에 적용할 색상
    const lines = selectedText.split('\n');

    const processMarkdown = (line) => {
        const markdownPatterns = [
            { pattern: /\*\*(.*?)\*\*/, replacement: `<span style='color: ${innerColor}'>$1</span>` }, // Bold
            { pattern: /__(.*?)__/, replacement: `<span style='color: ${innerColor}'>$1</span>` }, // Bold
            { pattern: /\*(.*?)\*/, replacement: `<span style='color: ${innerColor}'>$1</span>` }, // Italic
            { pattern: /_(.*?)_/, replacement: `<span style='color: ${innerColor}'>$1</span>` }, // Italic
            { pattern: /~~(.*?)~~/, replacement: `<span style='color: ${innerColor}'>$1</span>` }, // Strikethrough
            { pattern: /==(.*?)==/, replacement: `<span style='color: ${innerColor}'>$1</span>` }, // Highlight
            { pattern: /`(.*?)`/, replacement: `<span style='color: ${innerColor}'>$1</span>` }, // Inline code
            { pattern: /\[(.*?)\]\(.*?\)/, replacement: `<span style='color: ${innerColor}'>$1</span>` }, // Hyperlink
            { pattern: /\[\[(.*?)\]\]/, replacement: `<span style='color: ${innerColor}'>$1</span>` }, // Internal link
            { pattern: /!\[(.*?)\]\(.*?\)/, replacement: `<span style='color: ${innerColor}'>$1</span>` }, // Image
            { pattern: /\$\$(.*?)\$\$/, replacement: `<span style='color: ${innerColor}'>$1</span>` }, // Block Math
            { pattern: /\$(.*?)\$/, replacement: `<span style='color: ${innerColor}'>$1</span>` }, // Inline Math
            { pattern: /%%(.*?)%%/, replacement: `<span style='color: ${innerColor}'>$1</span>` }, // Comment
        ];

        markdownPatterns.forEach(({ pattern, replacement }) => {
            line = line.replace(pattern, replacement);
        });

        return line;
    };

    const processedLines = lines.map(line => {
        // 기존의 색상이 적용된 <span> 태그를 제거
        const lineWithoutSpan = line.replace(/<span style='color: [^']+'>(.*?)<\/span>/g, '$1');
        // 선행 공백을 유지한 채 줄 시작을 알파벳이나 숫자가 아닌 공백 이후의 텍스트로 바꿈
        const leadingSpaces = lineWithoutSpan.match(/^\s*/)[0];
        const lineContent = lineWithoutSpan.trim().replace(/^[-1.]+\s*/, '').replace(/^#+\s*/, '');
        const processedContent = processMarkdown(lineContent);
        return `${leadingSpaces}<span style='color: ${color}'>${processedContent}</span>`;
    });

    const coloredText = processedLines.join('\n');
    tR += coloredText;
} 
_%>
