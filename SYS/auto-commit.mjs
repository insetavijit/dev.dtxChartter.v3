#!/usr/bin/env node

import simpleGit from 'simple-git';
import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';

const git = simpleGit();

// Parse command-line arguments
const argv = yargs(hideBin(process.argv))
    .option('branch', {
        alias: 'b',
        type: 'string',
        default: 'main',
        description: 'Branch to commit and push to',
    })
    .option('message', {
        alias: 'm',
        type: 'string',
        description: 'Custom commit message (overrides timestamp)',
    })
    .option('remote', {
        alias: 'r',
        type: 'string',
        default: 'origin',
        description: 'Remote repository name',
    })
    .option('dry-run', {
        type: 'boolean',
        default: false,
        description: 'Simulate the Git operations without executing them',
    })
    .help()
    .parse();

// Generate timestamp for default commit message
const now = new Date();
const timestamp = now.toISOString().replace(/[:.]/g, '-'); // e.g., 2025-09-20T12-34-56-789Z
const commitMessage = argv.message || `Auto-commit: ${timestamp}`;

// Logger for consistent output
const log = (message, isError = false) => {
    const prefix = isError ? '❌' : 'ℹ️';
    console.log(`${prefix} ${message}`);
};

async function autoCommit() {
    try {
        // Check if in a Git repository
        if (!(await git.checkIsRepo())) {
            log('Not a Git repository. Initializing...');
            if (!argv['dry-run']) {
                await git.init();
                log('Git repository initialized');
            } else {
                log('Would initialize Git repository (dry run)');
            }
        }

        // Check for changes
        const status = await git.status();
        if (!status.files.length) {
            log('No changes to commit');
            return;
        }

        // Stage all changes
        log('Adding all changes...');
        if (!argv['dry-run']) {
            await git.add('./*');
        } else {
            log('Would stage all changes (dry run)');
        }

        // Commit changes
        log(`Committing with message: "${commitMessage}"`);
        if (!argv['dry-run']) {
            await git.commit(commitMessage);
        } else {
            log('Would commit changes (dry run)');
        }

        // Check if remote exists
        const remotes = await git.getRemotes(true);
        const remoteExists = remotes.some((r) => r.name === argv.remote);
        if (!remoteExists) {
            log(`Remote "${argv.remote}" not found. Skipping push.`);
            return;
        }

        // Ensure branch exists locally
        const branches = await git.branchLocal();
        if (!branches.all.includes(argv.branch)) {
            log(`Branch "${argv.branch}" does not exist locally. Creating...`);
            if (!argv['dry-run']) {
                await git.checkoutLocalBranch(argv.branch);
            } else {
                log(`Would create branch "${argv.branch}" (dry run)`);
            }
        }

        // Push to remote
        log(`Pushing to ${argv.remote}/${argv.branch}...`);
        if (!argv['dry-run']) {
            await git.push(argv.remote, argv.branch, { '--set-upstream': null });
        } else {
            log(`Would push to ${argv.remote}/${argv.branch} (dry run)`);
        }

        log('✅ Auto commit and push completed!');
    } catch (err) {
        log(`Git automation failed: ${err.message}`, true);
        process.exit(1);
    }
}

// Run the script
autoCommit();
