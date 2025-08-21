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
    let bizMessage: string | undefined;

    if (typeof err.response?.data?.detail === 'string') {
      bizMessage = err.response.data.detail;
    } else if (typeof err.response?.data?.detail.message === 'string') {
      bizMessage = err.response.data.detail.message;
    } else {
      bizMessage = err.response?.data?.message;
    }

    if (bizMessage) {
      toast.error(bizMessage);
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
