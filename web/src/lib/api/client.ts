'use client';

import {
  AuditApi,
  Configuration,
  DefaultApi,
  GraphApi,
  QuotasApi,
} from '@/api';
import axios from 'axios';
import { toast } from 'sonner';

const configuration = new Configuration();

const request = axios.create({
  baseURL: `/api/v1`,
  timeout: 1000 * 5,
});

request.interceptors.response.use(
  function (response) {
    // Any status code that lie within the range of 2xx cause this function to trigger
    // Do something with response data
    return response;
  },
  function (err: any) {
    const bizData = err.response?.data;
    if (bizData?.message) {
      toast.error(bizData.message);
    }
    return Promise.reject(err);
  },
);

export const apiClient = {
  defaultApi: new DefaultApi(configuration, undefined, request),
  graphApi: new GraphApi(configuration, undefined, request),
  quotasApi: new QuotasApi(configuration, undefined, request),
  auditApi: new AuditApi(configuration, undefined, request),
};
