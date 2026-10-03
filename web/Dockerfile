FROM node:22-alpine AS cesium
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --omit=dev

FROM nginx:1.27-alpine
COPY --from=cesium /app/node_modules/cesium/Build/Cesium /usr/share/nginx/html/cesium
COPY index.html app.js styles.css /usr/share/nginx/html/
COPY nginx.conf /etc/nginx/conf.d/default.conf
