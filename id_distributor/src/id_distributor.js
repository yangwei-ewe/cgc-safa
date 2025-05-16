import express from 'express';
import bodyParser from 'body-parser';
import cors from 'cors';

const app = express();
const buffer = new Uint32Array(1);
buffer[0] = 0;

app.use(bodyParser.json());
app.use(cors({
    origin: '*',
    methods: ['GET'],
    allowedHeaders: ['Content-Type', 'Authorization']
}));

app.get('/', (req, res) => {
    res.status(200).send('hello, world!');
});

app.get('/get', (req, res) => {
    res.status(200).send(Atomics.add(buffer, 0, 1));
})

app.get('/check', (req, res) => {
    res.status(200).send(Atomics.load(buffer, 0));
})

app.listen(8621, () => console.log('Server started on port 8621'))
