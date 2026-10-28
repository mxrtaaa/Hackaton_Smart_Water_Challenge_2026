
import http from 'node:http';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

import { getDashboard } from './data-source.mjs';
import { getModelCosts } from './real-data-source.mjs';


const root = path.resolve(
  fileURLToPath(new URL('./public/', import.meta.url))
);

const explicitPort = Boolean(process.env.PORT);
let port = Number(process.env.PORT || 4173);
const maxPort = port + 10;

if (!Number.isInteger(port) || port < 0 || port > 65535) {
  console.error('Niepoprawny PORT. Podaj liczbę całkowitą od 0 do 65535.');
  process.exit(1);
}

const types = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.svg': 'image/svg+xml'
};


const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, 'http://localhost');

  try {
    if (req.method === 'POST' && url.pathname === '/api/cost-scenarios') {
      let body = '';
      for await (const chunk of req) {
        body += chunk;
        if (Buffer.byteLength(body) > 16384) throw new Error('Zbyt duże żądanie.');
      }
      const input = JSON.parse(body);
      if ((!input.station || input.station === 'chorzow') && (process.env.UI_DATA_FILE || process.env.UI_SOURCE === 'demo')) {
        throw new Error('Wycena Python wymaga domyślnego źródła modeli i raportów.');
      }
      const result = await getModelCosts(input.station, input.days, input.prices, input.period, input.metric, {startDate:input.startDate,endDate:input.endDate,energyBasis:input.energyBasis});
      res.writeHead(200, {'Content-Type':'application/json; charset=utf-8','Cache-Control':'no-store'});
      res.end(JSON.stringify(result));
      return;
    }

    if (!['GET', 'HEAD'].includes(req.method)) {
      res.writeHead(405);
      res.end();
      return;
    }

    if (url.pathname === '/api/dashboard') {
      let data;

      if (process.env.UI_DATA_FILE && (!url.searchParams.get('station') || url.searchParams.get('station') === 'chorzow')) {
        const snapshot = JSON.parse(
          await readFile(process.env.UI_DATA_FILE, 'utf8')
        );

        data = snapshot.dashboards?.find(
          d => d.station.id === (
            url.searchParams.get('station') || 'chorzow'
          )
        );

        if (!data) {
          throw new Error('Brak obiektu w źródle JSON.');
        }
        // A saved snapshot is independent of the local Python reports.
        data = {...data, businessEngine: 'snapshot'};
      } else {
        data = await getDashboard(
          url.searchParams.get('station') || 'chorzow'
        );
      }

      res.writeHead(200, {
        'Content-Type': 'application/json; charset=utf-8',
        'Cache-Control': 'no-store'
      });

      res.end(JSON.stringify(data));
      return;
    }

    const file = path.resolve(
      root,
      '.' + decodeURIComponent(
        url.pathname === '/' ? '/index.html' : url.pathname
      )
    );

    if (!file.startsWith(root + path.sep)) {
      res.writeHead(403);
      res.end();
      return;
    }

    const body = await readFile(file);

    res.writeHead(200, {
      'Content-Type': types[path.extname(file)] || 'application/octet-stream',
      'X-Content-Type-Options': 'nosniff',
      'Cache-Control': 'no-store'
    });

    res.end(req.method === 'HEAD' ? undefined : body);

  } catch (error) {
    res.writeHead(
      url.pathname.startsWith('/api/') ? 400 : 404,
      {
        'Content-Type': 'application/json; charset=utf-8'
      }
    );

    res.end(
      JSON.stringify({
        error: url.pathname.startsWith('/api/')
          ? error.message
          : 'Nie znaleziono pliku.'
      })
    );
  }
});


server.on('error', error => {
  if (error.code === 'EADDRINUSE') {
    if (!explicitPort && port < maxPort) {
      console.log(`Port ${port} jest zajęty. Sprawdzam port ${port + 1}…`);
      port += 1;
      server.listen(port, '127.0.0.1');
      return;
    }

    console.error(`Nie można uruchomić panelu: port ${port} jest zajęty.`);
    console.error('Zatrzymaj poprzedni serwer przez Ctrl+C w jego terminalu lub ustaw inny PORT.');
  } else {
    console.error(`Nie można uruchomić panelu: ${error.message}`);
  }
  process.exitCode = 1;
});

server.on('listening', () => {
  console.log(`Smart Water UI: http://127.0.0.1:${server.address().port}`);
});

server.listen(port, '127.0.0.1');
