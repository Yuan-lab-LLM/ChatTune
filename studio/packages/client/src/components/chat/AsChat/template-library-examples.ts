import templatesData from './templates.json';

interface TemplateExample {
    id: string;
    content: string;
}

interface TemplateLibraryData {
    examples?: TemplateExample[];
}

const typedTemplatesData = templatesData as TemplateLibraryData;
const ADMIN_ONLY_EXAMPLE_IDS = new Set([
    'example-system-view',
    'example-gpu-status',
]);

export const getExampleTemplates = (isAdmin = true): string[] => {
    return (typedTemplatesData.examples || [])
        .filter((example) => isAdmin || !ADMIN_ONLY_EXAMPLE_IDS.has(example.id))
        .map((example) => example.content);
};

export const rotateExamples = (
    examples: string[],
    currentIndex: number,
): string => {
    if (examples.length === 0) return '';
    return examples[currentIndex % examples.length];
};
