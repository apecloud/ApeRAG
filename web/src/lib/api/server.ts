'use server';

import {
  AuditApi,
  ChatDocumentsApi,
  Configuration,
  DefaultApi,
  EvaluationApi,
  GraphApi,
  QuotasApi,
} from '@/api';
import { getLocale } from '@/services/cookies';
import axios from 'axios';
import { cookies } from 'next/headers';

const configuration = new Configuration();

const request = axios.create({
  baseURL:
    (process.env.API_SERVER_ENDPOINT || 'http://localhost:8000') +
    (process.env.API_SERVER_BASE_PATH || '/api/v1'),
  timeout: 1000 * 5,
});

request.interceptors.request.use(
  async (config) => {
    const lang = await getLocale();
    const allCookies = (await cookies())
      .getAll()
      .map((item) => `${item.name}=${item.value}`)
      .join('; ');
    Object.assign(config.headers, {
      Lang: lang,
      Cookie: allCookies,
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
  evaluationApi: new EvaluationApi(configuration, undefined, request),
  chatDocumentsApi: new ChatDocumentsApi(configuration, undefined, request),
};

export const getServerApi = async () => {
  return api;
};
