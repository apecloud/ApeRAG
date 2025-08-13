import { Configuration, DefaultApi, GraphApi, QuotasApi } from '@/api';
import axios from 'axios';


export const request = axios.create({
  baseURL: 'http://localhost:3000/api/v1',
  timeout: 1000 * 5,
});

const requestConfiguration = new Configuration();

export const defaultApi = new DefaultApi(requestConfiguration, undefined, request);
export const graphApi = new GraphApi(requestConfiguration, undefined, request);
export const quotasApi = new QuotasApi(requestConfiguration, undefined, request);