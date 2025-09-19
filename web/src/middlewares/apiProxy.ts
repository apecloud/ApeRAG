import { cookies } from 'next/headers';
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

    if (pathname.match(new RegExp('/api/v1'))) {
      const destination = new URL(host);
      const url = req.nextUrl.clone();
      url.host = destination.host;
      url.port = destination.port;

      if (process.env.NEXT_PUBLIC_BASE_PATH) {
        url.pathname = pathname.replace(
          process.env.NEXT_PUBLIC_BASE_PATH,
          apiServerBasePath,
        );
      }

      url.basePath = '';

      const response = NextResponse.rewrite(url);

      const allCookies = await cookies();
      const session = allCookies.get('session');

      if (session) {
        response.headers.set('Authorization', `Bearer ${session.value}`);
      }

      return response;
    } else {
      return next(req, event);
    }
  };
}
