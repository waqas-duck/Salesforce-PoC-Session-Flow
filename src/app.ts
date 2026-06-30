import type { Request, Response } from 'express';
import express from 'express';

const app = express();
app.use(express.json());

app.get('/health', (_: Request, res: Response) => {
    res.status(200).json({ status: 'healthy' });
});

app.get('/', (_: Request, res: Response) => {
    res.status(200).json({ message: '{{PROJECT_NAME}} is running' });
});

export default app;
