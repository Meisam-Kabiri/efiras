// Test file for parsing logic matching our frontend React code
const testText = `### Data Processing Agreements

GDPR **Article 28** establishes the core requirement that a formal contract must exist (1).

- First requirement of **Article 28** (1)
- Second requirement (2)
- Citation group (1)(2)

Sources:
1. GDPR, Article 28, Page 12`;

const getSourcesFromContent = (content) => {
  const sources = [];
  if (content && content.includes('Sources:')) {
    const sourcesSection = content.split('Sources:')[1];
    if (sourcesSection) {
      const lines = sourcesSection.split('\n').filter(line => line.trim());
      lines.forEach(line => {
        const cleanLine = line.replace(/^[-*•\d+\.\)\s]+/, '').trim();
        if (cleanLine && cleanLine.length > 10 && !cleanLine.toLowerCase().includes('note:')) {
          sources.push(cleanLine);
        }
      });
    }
  }
  return sources;
};

const renderCitations = (text) => {
  const citationRegex = /(\(\d+\))/g;
  const parts = text.split(citationRegex);
  return parts.map((part) => {
    if (part.match(/^\(\d+\)$/)) {
      const num = part.slice(1, -1);
      return `[Citation ${num}]`;
    }
    return part;
  });
};

const renderTextWithBoldAndCitations = (text) => {
  const parts = text.split(/(\*\*.*?\*\*)/g);
  return parts.map((part) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      const boldContent = part.slice(2, -2);
      return `[BOLD: ${renderCitations(boldContent).join('')}]`;
    }
    return renderCitations(part).join('');
  });
};

const parseMarkdownAndCitations = (text) => {
  const lines = text.split("\n");
  return lines.map((line, lineIdx) => {
    const cleanLine = line.trim();
    if (!cleanLine) return `[EMPTY LINE]`;

    if (cleanLine.startsWith("###")) {
      return `[H3: ${renderTextWithBoldAndCitations(cleanLine.replace(/^###\s*/, "")).join('')}]`;
    }

    if (cleanLine.startsWith("-") || cleanLine.startsWith("•") || cleanLine.startsWith("*")) {
      return `[BULLET: ${renderTextWithBoldAndCitations(cleanLine.replace(/^[-•*]\s*/, "")).join('')}]`;
    }

    return `[P: ${renderTextWithBoldAndCitations(cleanLine).join('')}]`;
  });
};

console.log("--- Extracted Sources ---");
const sources = getSourcesFromContent(testText);
console.log(sources);

console.log("\n--- Parsed Text Content ---");
const contentOnly = testText.split('Sources:')[0].trim();
const parsed = parseMarkdownAndCitations(contentOnly);
parsed.forEach(p => console.log(p));
