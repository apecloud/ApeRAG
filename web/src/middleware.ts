import { NextMiddleware, NextResponse } from 'next/server';
import { withApiProxy } from './middlewares/apiProxy';

type MiddlewareFactory = (next: NextMiddleware) => NextMiddleware;

const stackMiddlewares = (
  functions: MiddlewareFactory[],
  index = 0,
): NextMiddleware => {
  const fn = functions[index];
  if (fn) {
    const next = stackMiddlewares(functions, index + 1);
    return fn(next);
  }
  return () => NextResponse.next();
};

export default stackMiddlewares([withApiProxy]);

// Routes Middleware should not run on
export const config = {
  matcher: ['/((?!static|.*\\..*|_next|favicon.ico|robots.txt).*)'],
};
