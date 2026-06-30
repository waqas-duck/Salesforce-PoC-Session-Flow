# Builder stage — compile TypeScript
FROM store.sfdcbt.net/node:22 AS builder

WORKDIR /app

ENV NPM_TOKEN=""
COPY .npmrc package.json ./
RUN npm install

COPY tsconfig.json ./
COPY src ./src
RUN npm run build

# Production stage — minimal runtime
FROM store.sfdcbt.net/node:22-alpine

WORKDIR /app

ENV NPM_TOKEN=""
COPY .npmrc package.json ./
RUN npm install --omit=dev

COPY --from=builder /app/dist ./dist

RUN apk update && apk add --no-cache curl && apk upgrade --no-cache zlib

RUN addgroup -g 1001 -S nodejs && \
    adduser -S nodejs -u 1001 && \
    chown -R nodejs:nodejs /app

RUN rm -rf /usr/local/lib/node_modules/npm /usr/local/bin/npm /usr/local/bin/npx || true
USER nodejs

HEALTHCHECK --interval=30s --timeout=3s --start-period=40s \
  CMD node -e "require('http').get('http://localhost:3000/health', (r) => process.exit(r.statusCode === 200 ? 0 : 1)).on('error', () => process.exit(1)).end()"

EXPOSE 3000

CMD ["node", "dist/server.js"]
