import { compare } from 'semver';
import packageJson from '../../../../package.json';

/**
 * Minimal npm package info used for version comparison.
 */
interface NpmPackageInfo {
    'dist-tags': {
        latest: string;
    };
}

/**
 * Check whether a newer version of the app is available on npm.
 *
 * - Fetches the latest dist-tag for `@medflow/studio` through the Studio server
 * - Compares it with the current bundled version
 *
 * @returns If newer: { hasUpdate: true, currentVersion, latestVersion };
 *          otherwise: { hasUpdate: false, currentVersion, latestVersion };
 *          on error: { hasUpdate: false, error: true }.
 */
export async function checkForUpdates() {
    try {
        const response = await fetch(
            '/api/version-check',
        );
        if (!response.ok) {
            throw new Error(`Version check failed with status ${response.status}`);
        }
        const data: NpmPackageInfo = await response.json();
        const latestVersion = data['dist-tags'].latest;
        const currentVersion = packageJson.version;

        // Check if the latest version is greater than the current version
        if (compare(latestVersion, currentVersion) === 1) {
            return {
                hasUpdate: true,
                currentVersion,
                latestVersion,
            };
        }

        return {
            hasUpdate: false,
            currentVersion,
            latestVersion,
        };
    } catch (error) {
        console.error('Failed to check for updates:', error);
        return {
            hasUpdate: false,
            error: true,
        };
    }
}
