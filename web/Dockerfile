# Build stage
FROM node:18-alpine as builder

WORKDIR /app

# Copy package files
COPY package.json yarn.lock ./

# Install dependencies with retry and timeout settings
RUN yarn config set registry https://registry.npmmirror.com && \
    yarn config set network-timeout 300000 && \
    yarn install --network-timeout 300000 --retry 10

# Copy source code
COPY . .

# Build the application
RUN yarn build

# Production stage
FROM nginx:mainline-alpine3.18-slim

# Copy built files from builder stage
COPY --from=builder /app/build /html/web

COPY ./deploy/startup.sh /docker-entrypoint.d/

EXPOSE 80