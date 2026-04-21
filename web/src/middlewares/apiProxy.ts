import {
  NextFetchEvent,
  NextMiddleware,
  NextRequest,
  NextResponse,
} from 'next/server';

export function withApiProxy(next: NextMiddleware): NextMiddleware {
  return async (req: NextRequest, event: NextFetchEvent) => {
    const { pathname } = req.nextUrl;

    const host = process.env.API_SERVER_ENDPOINT || 'http://localhost:8000';
    const apiServerBasePath = process.env.API_SERVER_BASE_PATH || '/api/v1';
    const apiPathMatch = pathname.match(/\/api\/v(1|2)(\/.*)?$/);

    if (apiPathMatch) {
      const destination = new URL(host);
      const url = req.nextUrl.clone();
      url.host = destination.host;
      url.port = destination.port;

      if (process.env.NEXT_PUBLIC_BASE_PATH) {
        const requestedApiPrefix = `/api/v${apiPathMatch[1]}`;
        const proxyBasePath =
          requestedApiPrefix === '/api/v1'
            ? apiServerBasePath
            : apiServerBasePath.replace(/\/api\/v1$/, requestedApiPrefix);
        url.pathname = pathname.replace(
          process.env.NEXT_PUBLIC_BASE_PATH,
          proxyBasePath,
        );
      }

      url.basePath = '';

      return NextResponse.rewrite(url);
    } else {
      return next(req, event);
    }
  };
}
