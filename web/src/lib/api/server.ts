'use server';

import {
  AuditApi,
  Configuration,
  DefaultApi,
  GraphApi,
  QuotasApi,
} from '@/api';
import { getCookie, getLocale } from '@/services/cookies';
import axios from 'axios';

const configuration = new Configuration();

const request = axios.create({
  baseURL: `${process.env.API_ENDPOINT}/api/v1`,
  timeout: 1000 * 5,
});

request.interceptors.request.use(
  async (config) => {
    const lang = await getLocale();
    const session = await getCookie('session');
    Object.assign(config.headers, {
      Lang: lang,
      Cookie: `session=${session}`,
    });
    return config;
  },
  function (error) {
    return Promise.reject(error);
  },
);

const api = {
  defaultApi: new DefaultApi(configuration, undefined, request),
  graphApi: new GraphApi(configuration, undefined, request),
  quotasApi: new QuotasApi(configuration, undefined, request),
  auditApi: new AuditApi(configuration, undefined, request),
};

export const getServerApi = async () => {
  return api;
};
