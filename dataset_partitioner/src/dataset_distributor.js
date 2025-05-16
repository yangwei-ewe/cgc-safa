import express from 'express';
import path from 'path';
import cors from 'cors';

const app = express();

app.use(cors({
  origin: '*',
  methods: ['GET'],
  allowedHeaders: ['Content-Type']
}));

const DIR = '/data';
const PORT = 3000;

app.get('/ready', (req, res) => {
  res.status(200).send('OK');
});

app.get('/ds/:p', (req, res) => {
  const f = path.join(DIR, `${req.params.p}.npz`);
  res.sendFile(f, err => {
    if (err) res.status(404).send(`Partition ${req.params.p} not found`);
  });
});

app.listen(PORT, () => {
  console.log(`Distributor listening on ${PORT}, serving from ${DIR}`);
});
