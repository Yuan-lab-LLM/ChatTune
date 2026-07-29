/**
 * Display a startup banner for MedFlow ChatTune
 * Inspired by Phoenix's banner style
 */

import chalk from 'chalk';
import figlet from 'figlet';

export function displayBanner(
    appName: string,
    version: string,
    port: number,
    otelGrpcPort: number,
    databasePath: string,
    mode: 'development' | 'production',
    publicUrl?: string,
): void {
    const studioUrl = publicUrl || `http://localhost:${port}`;
    let publicHost = 'localhost';
    try {
        publicHost = new URL(studioUrl).hostname;
    } catch {
        // Keep the local fallback for malformed deployment configuration.
    }
    // Create welcome message with border
    const welcomeMessage = `* Welcome to ${chalk.bold('MedFlow ChatTune')} v${version}! *`;
    // Strip ANSI codes to calculate the actual display length
    const displayLength =
        // eslint-disable-next-line no-control-regex
        welcomeMessage.replace(/\u001b\[[0-9;]*m/g, '').length;
    const borderLength = displayLength + 4;
    const topBorder = '┌' + '─'.repeat(borderLength - 2) + '┐';
    const bottomBorder = '└' + '─'.repeat(borderLength - 2) + '┘';
    const welcomeBanner = `${topBorder}\n│ ${welcomeMessage} │\n${bottomBorder}`;

    // Generate ASCII art using figlet
    let asciiText: string;
    try {
        asciiText = figlet.textSync(appName, {
            font: 'ANSI Shadow',
            horizontalLayout: 'default',
            verticalLayout: 'default',
        });
    } catch {
        asciiText = appName;
    }

    if (!asciiText || asciiText.trim().length === 0) {
        asciiText = appName;
    }

    // Wrap ASCII art in a border
    const lines = asciiText
        .split('\n')
        .filter((line) => line.trim().length > 0);
    if (lines.length === 0) {
        lines.push(appName);
    }

    const appNameBanner = [...lines].join('\n');

    // Community and documentation links with colors
    const modeColor = mode === 'production' ? chalk.green : chalk.yellow;
    const links = `
${chalk.cyan('🌍  Join our Community  🌍')}
${chalk.blue('https://github.com/MedFlow2025/medflow')}

${chalk.yellow('⭐  Leave us a Star  ⭐')}
${chalk.blue('https://github.com/MedFlow2025/medflow')}

${chalk.magenta('📚  Documentation  📚')}
${chalk.blue('https://github.com/MedFlow2025/medflow')}

    ${chalk.green('🚀  MedFlow ChatTune Server  🚀')}
    ${chalk.bold('Studio UI:')}      ${chalk.cyan(studioUrl)}
    ${chalk.bold('Traces Endpoint:')}
    ${chalk.bold('  - HTTP:')}       ${chalk.cyan(`${studioUrl.replace(/\/+$/, '')}/v1/traces`)}
    ${chalk.bold('  - gRPC:')}       ${chalk.cyan(`http://${publicHost}:${otelGrpcPort}`)}
    ${chalk.bold('Mode:')}           ${modeColor(mode)}
    ${chalk.bold('Storage:')}        ${chalk.gray(databasePath)}
`;

    // Display banner with a separator line
    console.log('');
    console.log(chalk.cyan(welcomeBanner));
    console.log('');
    console.log(chalk.cyan(appNameBanner));
    console.log(links);
    console.log('');
}
