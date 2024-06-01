<%_* 
const selectedText = tp.file.selection();
if (selectedText) {
    const color = '#f4acb6'; // 변경할 색상
    const lines = selectedText.split('\n');

    const processMarkdown = (line) => {
        const markdownPatterns = [
            { pattern: /\*\*(.*?)\*\*/, replacement: `<span style='color: ${color}'>$1</span>` }, // Bold
            { pattern: /__(.*?)__/, replacement: `<span style='color: ${color}'>$1</span>` }, // Bold
            { pattern: /\*(.*?)\*/, replacement: `<span style='color: ${color}'>$1</span>` }, // Italic
            { pattern: /_(.*?)_/, replacement: `<span style='color: ${color}'>$1</span>` }, // Italic
            { pattern: /~~(.*?)~~/, replacement: `<span style='color: ${color}'>$1</span>` }, // Strikethrough
            { pattern: /==(.*?)==/, replacement: `<span style='color: ${color}'>$1</span>` }, // Highlight
            { pattern: /`(.*?)`/, replacement: `<span style='color: ${color}'>$1</span>` }, // Inline code
            { pattern: /\[(.*?)\]\(.*?\)/, replacement: `<span style='color: ${color}'>$1</span>` }, // Hyperlink
            { pattern: /\[\[(.*?)\]\]/, replacement: `<span style='color: ${color}'>$1</span>` }, // Internal link
            { pattern: /!\[(.*?)\]\(.*?\)/, replacement: `<span style='color: ${color}'>$1</span>` }, // Image
            { pattern: /\$\$(.*?)\$\$/, replacement: `<span style='color: ${color}'>$1</span>` }, // Block Math
            { pattern: /\$(.*?)\$/, replacement: `<span style='color: ${color}'>$1</span>` }, // Inline Math
            { pattern: /%%(.*?)%%/, replacement: `<span style='color: ${color}'>$1</span>` }, // Comment
        ];

        markdownPatterns.forEach(({ pattern, replacement }) => {
            line = line.replace(pattern, replacement);
        });

        return line;
    };

    let isFirstLine = true;

    const processedLines = lines.map(line => {
        // 기존의 색상이 적용된 <span> 태그를 제거
        let lineWithoutSpan = line.replace(/<span style='color: [^']+'>(.*?)<\/span>/g, '$1');

        // 첫 번째 줄인 경우
        if (isFirstLine) {
            isFirstLine = false;
            // 선행 공백과 기호를 유지하면서 텍스트 부분을 추출
            const leadingSpacesAndDash = lineWithoutSpan.match(/^(\s*-?\s*)/)[0];
            const lineContent = lineWithoutSpan.slice(leadingSpacesAndDash.length);
            const processedContent = processMarkdown(lineContent);
            return `${leadingSpacesAndDash}<span style='color: ${color}'>${processedContent}</span>`;
        } else {
            // 선행 공백과 기호를 유지하면서 텍스트 부분을 추출
            const leadingSpacesAndDash = lineWithoutSpan.match(/^(\s*-?\s*)/)[0];
            const lineContent = lineWithoutSpan.slice(leadingSpacesAndDash.length);
            const processedContent = processMarkdown(lineContent);
            return `${leadingSpacesAndDash}<span style='color: ${color}'>${processedContent}</span>`;
        }
    });

    const coloredText = processedLines.join('\n');
    tR += coloredText;
} 
_%>
