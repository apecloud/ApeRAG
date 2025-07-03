import { Configuration, DefaultApi, GraphApi } from '@/api';
import { getAuthorizationHeader } from '@/models/user';
import axios from 'axios';
import { toast } from 'react-toastify';
import { getLocale, history } from 'umi';

export const request = axios.create({
  baseURL: '',
  timeout: 1000 * 5,
});

request.interceptors.request.use(
  (config) => {
    const lang = getLocale();
    Object.assign(config.headers, {
      lang,
      ...getAuthorizationHeader(),
    });
    config.headers['lang'] = lang;
    return config;
  },
  function (error) {
    // Do something with request error
    return Promise.reject(error);
  },
);

request.interceptors.response.use(
  function (response) {
    // Any status code that lie within the range of 2xx cause this function to trigger
    // Do something with response data

    return response;
  },
  function (err: any) {
    // Any status codes that falls outside the range of 2xx cause this function to trigger
    // Do something with response error
    if (err.status === 401) {
      history.replace(
        `/accounts/signin?redirectUri=${encodeURIComponent(
          window.location.href,
        )}`,
      );
    } else {
      toast.error(err.response.data?.detail || err.message || 'request error');
    }
    return Promise.reject(err);
  },
);

const requestConfiguration = new Configuration();

export const api = new DefaultApi(requestConfiguration, '/api/v1', request);
export const graphApi = new GraphApi(requestConfiguration, '/api/v1', request);
