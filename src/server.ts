import app from './app';

const PORT = process.env.PORT ?? 3000;

const server = app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});

process.on('SIGTERM', () => {
    server.close(() => process.exit(0));
});
