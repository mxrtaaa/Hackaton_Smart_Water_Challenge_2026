import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { createInterface } from 'node:readline';

const backend = fileURLToPath(new URL('./model_backend.py', import.meta.url));

let worker;
let nextId = 0;

const pending = new Map();

function pythonPath() {
    if (process.env.UI_PYTHON) {
        return process.env.UI_PYTHON;
    }

    const portable = fileURLToPath(
        new URL('./.runtime/python/python.exe', import.meta.url)
    );

    if (existsSync(portable)) {
        return portable;
    }

    return process.platform === 'win32'
        ? fileURLToPath(new URL('../.venv/Scripts/python.exe', import.meta.url))
        : 'python3';
}

function failAll(error) {
    for (const p of pending.values()) {
        clearTimeout(p.timeout);
        p.reject(error);
    }

    pending.clear();
}

function ensureWorker() {
    if (worker) {
        return;
    }

    const child = spawn(
        pythonPath(),
        ['-X', 'utf8', '-u', backend],
        {
            windowsHide: true,
            stdio: ['pipe', 'pipe', 'pipe']
        }
    );

    worker = child;

    let diagnostics = '';

    child.stderr.on('data', b => {
        diagnostics = (diagnostics + b.toString('utf8')).slice(-6000);
    });

    createInterface({ input: child.stdout }).on('line', line => {
        let response;

        try {
            response = JSON.parse(line);
        } catch {
            diagnostics = (diagnostics + '\n' + line).slice(-6000);
            return;
        }

        const p = pending.get(response.id);

        if (!p) {
            return;
        }

        clearTimeout(p.timeout);
        pending.delete(response.id);

        if (response.error) {
            p.reject(new Error(response.error));
        } else {
            p.resolve(response.result);
        }
    });

    child.on('error', error => {
        failAll(
            new Error(
                `Nie można uruchomić Pythona: ${error.message}. Ustaw UI_PYTHON na działający interpreter.`
            )
        );
    });

    child.on('close', () => {
        if (worker === child) {
            worker = null;
        }

        failAll(
            new Error(
                `Backend Python zakończył działanie. ${diagnostics || 'Sprawdź interpreter i wymagane biblioteki.'}`
            )
        );
    });

    child.stdin.on('error', error => {
        failAll(error);
    });
}

export function callPython(request) {
    ensureWorker();

    const id = ++nextId;

    return new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
            pending.delete(id);
            reject(new Error('Przekroczono czas oczekiwania na modele (120 s).'));
        }, 120000);

        pending.set(id, {
            resolve,
            reject,
            timeout
        });

        worker.stdin.write(
            JSON.stringify({ ...request, id }) + '\n'
        );
    });
}

export function stopPython() {
    if (worker) {
        worker.kill();
        worker = null;
    }
}

process.on('exit', stopPython);

for (const signal of ['SIGINT', 'SIGTERM']) {
    process.once(signal, () => {
        stopPython();
        process.exit(0);
    });
}