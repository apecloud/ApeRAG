FROM nginx:mainline-alpine3.18-slim

COPY ./dist /html/web

# Copy nginx startup script
COPY ./deploy/startup.sh /docker-entrypoint.d/

# Make startup script executable and clean up
RUN chmod +x /docker-entrypoint.d/startup.sh

EXPOSE 3000

CMD ["nginx", "-g", "daemon off;"] 