import type { NextApiRequest, NextApiResponse } from "next";
import type { NextHttpProxyMiddlewareOptions } from "next-http-proxy-middleware";
import httpProxyMiddleware from "next-http-proxy-middleware";
import { NextFetchEvent, NextMiddleware, NextRequest } from "next/server";

const handleProxyInit: NextHttpProxyMiddlewareOptions["onProxyInit"] = (
  proxy
) => {
  /**
   * Check the list of bindable events in the `http-proxy` specification.
   * @see https://www.npmjs.com/package/http-proxy#listening-for-proxy-events
   */
  proxy.on("proxyReq", (proxyReq, req, res) => {
    // ...
  });
  proxy.on("proxyRes", (proxyRes, req, res) => {
    // ...
  });
};

export function ApiProxy(next: NextMiddleware): NextMiddleware {
  return async (req: NextRequest, event: NextFetchEvent) => {
    // 1. Check if the current route is protected or public
    const path = req.nextUrl.pathname;
    if(path.match(/^\/api\/v1/)) {
      httpProxyMiddleware(req, res, {
        target: "http://example.com",
        onProxyInit: handleProxyInit,
      })
    }

    return next(req, event);
  };
}


// async (req: NextApiRequest, res: NextApiResponse) =>
//     httpProxyMiddleware(req, res, {
//       target: "http://example.com",
//       onProxyInit: handleProxyInit,
//     });