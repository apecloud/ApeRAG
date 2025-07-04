#!/bin/sh
# vim:sw=4:ts=4:et

API_HOST=${APERAG_CONSOLE_SERVICE_HOST:-"127.0.0.1"}
API_PORT=${APERAG_CONSOLE_SERVICE_PORT:-"8000"}

cat > /etc/nginx/nginx.conf << EOF
events {                                                                                               
    worker_connections  1024;                                                                                                                        
} 

http {                                                                                                                                               
    include       /etc/nginx/mime.types;                                                                                                             
    default_type  application/octet-stream;                                                                                                          
                                                                                                                                                     
    log_format  main  '\$remote_addr - \$remote_user [\$time_local] "\$request" '                                                                        
                      '\$status \$body_bytes_sent "\$http_referer" '                                                                                    
                      '"\$http_user_agent" "\$http_x_forwarded_for"';                                                                                  
                                                                                                                                                     
    access_log  /var/log/nginx/access.log  main;                                                                                                     
                                                                                                                                                     
    sendfile        on;                                                                                                                              
    #tcp_nopush     on;                                                                                                                              
                                                                                                                                                     
    keepalive_timeout  65;                                                                                                                                                                                                                                                            
                                                                                                                                                     
    # include /etc/nginx/conf.d/*.conf;

    gzip on;
    gzip_comp_level 5;
    gzip_min_length 1k;
    gzip_buffers 4 16k;
    gzip_proxied any;
    gzip_vary on;
    gzip_types
      application/javascript
      application/x-javascript
      text/javascript
      text/css
      text/xml
      application/xhtml+xml
      application/xml
      application/atom+xml
      application/rdf+xml
      application/rss+xml
      application/geo+json
      application/json
      application/ld+json
      application/manifest+json
      application/x-web-app-manifest+json
      image/svg+xml
      text/x-cross-domain-policy;
    gzip_static on;  
    gzip_disable "MSIE [1-6]\.";

    map \$http_upgrade \$connection_upgrade {
        default upgrade;
        ''      close;
    }
    server {                                                                                  
        listen       3000;                                                                                                                          
        listen  [::]:3000;                                                                                                                          
        server_name  localhost;
        autoindex on;
        client_max_body_size 2G;
        
        # API proxy
        location /api/ {
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_http_version 1.1;
            proxy_set_header Upgrade \$http_upgrade;
            proxy_set_header Connection \$connection_upgrade;
            proxy_pass http://$API_HOST:$API_PORT;
        }

        # Handle root path - redirect to /web/
        location = / {
            return 301 \$scheme://\$http_host/web/;
        }
        
        # Handle /web/ path - serve static files from /html/web
        location /web/ {
            root   /html;
            index  index.html index.htm;
            try_files \$uri \$uri/ /web/index.html;
        }
    }
}
EOF