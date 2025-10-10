#!/usr/bin/env node
import path from 'path';
import fs from 'fs-extra';

console.log("Current working directory:", process.cwd());


/**
 * Simple cp CLI
 * Usage: node cp.js <source> <destination>
 */
async function main() {
    const args = process.argv.slice(2);
    if (args.length < 2) {
        console.error('Usage: node cp.js <source> <destination>');
        process.exit(1);
    }

    const [sourceArg, destArg] = args;

    const srcPath = path.isAbsolute(sourceArg)
        ? sourceArg
        : path.resolve(process.cwd(), sourceArg);

    const destPath = path.isAbsolute(destArg)
        ? destArg
        : path.resolve(process.cwd(), destArg);

    try {
        await fs.copy(srcPath, destPath, { overwrite: true });
        console.log(`Copied ${srcPath} → ${destPath}`);
    } catch (err) {
        console.error('Error copying:', err.message);
        process.exit(1);
    }
}

main();
