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
    const basePath = process.env.API_SERVER_BASE_PATH || '/api/v1';

    const apiPrefix = `${process.env.NEXT_PUBLIC_BASE_PATH || ''}/api/v1`;

    if (pathname.match(new RegExp(apiPrefix))) {
      const destination = new URL(host + basePath);
      const url = req.nextUrl.clone();
      url.host = destination.host;
      url.port = destination.port;
      url.pathname = pathname.replace(apiPrefix, basePath);
      return NextResponse.rewrite(url);
    } else {
      return next(req, event);
    }
  };
}
